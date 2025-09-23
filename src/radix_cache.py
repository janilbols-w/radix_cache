from __future__ import annotations

"""
The radix tree data structure for managing the KV cache.
"""

import heapq
import time
from collections import defaultdict
from functools import partial
from typing import TYPE_CHECKING, List, Optional, Tuple

import torch

class TreeNode:

    counter = 0

    def __init__(self, id: Optional[int] = None, 
                 create_time: float = None, 
                 last_access_time: float = None,
                 hit_count: int = 0):
        self.children = defaultdict(TreeNode)
        self.parent = None
        self.key = None
        self.value = None
        self.lock_ref = 0
        self.create_time = create_time if create_time else time.time()
        self.last_access_time = last_access_time if last_access_time else time.time()

        self.hit_count = hit_count
        # indicating the node is loading KV cache from host
        self.loading = False
        # store the host indices of KV cache
        self.host_value = None

        self.id = TreeNode.counter if id is None else id
        TreeNode.counter += 1

    @property
    def evicted(self):
        return self.value is None

    @property
    def backuped(self):
        return self.host_value is not None

    def __lt__(self, other: "TreeNode"):
        return self.last_access_time < other.last_access_time


def _key_match_page_size1(key0: List, key1: List):
    i = 0
    for k0, k1 in zip(key0, key1):
        if k0 != k1:
            break
        i += 1
    return i


def _key_match_paged(key0: List, key1: List, page_size: int):
    min_len = min(len(key0), len(key1))

    i = 0
    while i < min_len:
        if key0[i : i + page_size] != key1[i : i + page_size]:
            break
        i += page_size

    return i

class MetricsTracker():
    def __init__(self):
        self.init_time = None
        self.reset()
    
    def reset(self):
        self.num_queries = 0 # total tok/blocks is queried by match_prefix
        self.num_hits = 0 # total tok/blocks hits is returned from match_prefix
        self.num_actives = 0
        self.num_stores = 0
        self.init_time = self.init_time if self.init_time else None
        self.retrieve_time_list = []
        self.max_retrieve_time = 0.0
        self.store_duration_list = []
        self.max_store_duration = 0.0

    def show_metrics(self):
        print("="*50)
        print("num_queries = {}".format(self.num_queries))
        print("num_hits = {}".format(self.num_hits))
        print("hit_rate = {:0.2f}%".format(self.num_hits/ self.num_queries *100 if self.num_queries else 0))
        print("num_actives = {}".format(self.num_actives))
        print("num_stores = {}".format(self.num_stores))
        print("active_rate = {:0.2f}%".format(self.num_actives / self.num_stores * 100 if self.num_stores else 0))
        print("retrieve_time_list = {}".format(self.retrieve_time_list[:10]))
        print("max_retrieve_time = {}".format(self.max_retrieve_time))
        print("store_duration_list = {}".format(self.store_duration_list[:10]))
        print("max_store_duration = {}".format(self.max_store_duration))

    def update_hits(self, num_queries: int, 
                    num_hits: int, 
                    retrieve_time: float = None):
        # triggered when prefix match
        self.num_queries += num_queries
        self.num_hits += num_hits
        # print("hit {} out of {} ({:0.1f}%)".format(num_hits, 
        #                                           num_queries, 
        #                                           num_hits/num_queries*100))
        if retrieve_time is not None:
            self.update_retrieve_time(retrieve_time)
    
    def update_retrieve_time(self, retrieve_time: float):
        self.retrieve_time_list.append(retrieve_time)
        if self.max_retrieve_time < retrieve_time:
            self.max_retrieve_time = retrieve_time

    def set_retrieve_time(self, retrieve_time_list: List):
        self.retrieve_time_list = retrieve_time_list
        self.max_retrieve_time = max(self.retrieve_time_list)

    
    def update_stores(self, num_stores: int, timestamp: float):
        # triggered when instert
        self.num_stores += num_stores
        if not self.init_time:
            self.init_time = timestamp
        self.init_time = min(self.init_time, timestamp)
        store_dur = timestamp - self.init_time
        if store_dur > self.max_store_duration:
            self.max_store_duration = store_dur

    def set_store_duration(self, store_duration_list: List):
        self.store_duration_list = store_duration_list
        self.max_store_duration = max(self.store_duration_list)
    
    def set_active_and_store(self, num_actives, num_stores):
        self.num_actives = num_actives
        self.num_stores = num_stores

class RadixCache():
    def __init__(
        self,
        page_size: int,
        disable: bool = False,
        timestamp: float = None
    ):
        self.page_size = page_size
        self.disable = disable

        self.device = torch.device("cpu")

        if self.page_size == 1:
            self.key_match_fn = _key_match_page_size1
            self.get_child_key_fn = lambda key: key[0]
        else:
            self.key_match_fn = partial(_key_match_paged, page_size=page_size)
            self.get_child_key_fn = lambda key: tuple(key[:page_size])
        self.reset(timestamp=timestamp)
        self.metrics = MetricsTracker()

    ##### Public API #####

    def reset(self, timestamp: float = None):
        self.root_node = TreeNode(create_time=timestamp, 
                                  last_access_time=timestamp)
        self.root_node.key = []
        self.root_node.value = []
        self.root_node.lock_ref = 1
        self.evictable_size_ = 0
        self.protected_size_ = 0

    def match_prefix(self, key: List[int], timestamp: float = None, **kwargs) -> Tuple[torch.Tensor, int]:
        """Find the matching prefix from the radix tree.
        Args:
            key: A list of token/blocks IDs to find a matching prefix.
        Returns:
            A tuple of a tensor of matching prefix token/blocks IDs and
            the last node that contains the prefix values. Note that
            this API can modify the internal state of the Radix tree.
            The last node create a new child if the prefix is shorter
            than the last node's value.
        """
        if self.disable or len(key) == 0:
            return (
                torch.empty(
                    (0,),
                    dtype=torch.int64,
                    device=self.device,
                ),
                self.root_node,
            )

        if self.page_size != 1:
            page_aligned_len = len(key) // self.page_size * self.page_size
            key = key[:page_aligned_len]

        value, last_node = self._match_prefix_helper(self.root_node, key, timestamp)
        if value:
            if torch.is_tensor(value):
                value = torch.cat(value)
            else:
                val = []
                for v in value:
                    val += v
                value = val
        else:
            value = torch.empty((0,), dtype=torch.int64, device=self.device)
        return value, last_node

    def insert(self, key: List, value=None, timestamp: float = None):
        if self.disable:
            return 0

        if value is None:
            value = [x for x in key]
        return self._insert_helper(self.root_node, key, value, timestamp)


    def pretty_print(self):
        self._print_helper(self.root_node, 0)
        print(f"#tokens/blocks: {self.total_size()}")

    def total_size(self):
        return self._total_size_helper()

    def evict(self, num_elemens: int):
        if self.disable:
            return

        leaves = self._collect_leaves()
        heapq.heapify(leaves)

        num_evicted = 0
        while num_evicted < num_elemens and len(leaves):
            x = heapq.heappop(leaves)

            if x == self.root_node:
                break
            if x.lock_ref > 0:
                continue

            # clear up buff
            # self.kv_pool_allocator.free(x.value)
            num_evicted += len(x.value)
            self._delete_leaf(x)

            if len(x.parent.children) == 0:
                heapq.heappush(leaves, x.parent)

    def inc_lock_ref(self, node: TreeNode):
        if self.disable:
            return 0

        delta = 0
        while node != self.root_node:
            if node.lock_ref == 0:
                self.evictable_size_ -= len(node.value)
                self.protected_size_ += len(node.value)
                delta -= len(node.value)
            node.lock_ref += 1
            node = node.parent
        return delta

    def dec_lock_ref(self, node: TreeNode):
        if self.disable:
            return 0

        delta = 0
        while node != self.root_node:
            if node.lock_ref == 1:
                self.evictable_size_ += len(node.value)
                self.protected_size_ -= len(node.value)
                delta += len(node.value)
            node.lock_ref -= 1
            node = node.parent
        return delta

    def evictable_size(self):
        return self.evictable_size_

    def protected_size(self):
        # protected size refers to the size of the cache that is locked
        return self.protected_size_

    def all_values_flatten(self):
        values = []

        def _dfs_helper(node: TreeNode):
            for _, child in node.children.items():
                values.append(child.value)
                _dfs_helper(child)

        _dfs_helper(self.root_node)
        return torch.cat(values)

    def show_metrics(self):
        self.metrics.show_metrics()

    def update_cache_status(self, cur_time: float = None):
        if cur_time is None:
            cur_time = time.time()
        nodes = self._collect_nodes()
        # retrieve time
        retrieve_time_list = [node.last_access_time - node.create_time for node in nodes]
        self.metrics.set_retrieve_time(retrieve_time_list)
        # store duration
        store_duration_list = [cur_time - node.create_time for node in nodes]
        self.metrics.set_store_duration(store_duration_list)
        # num_stores
        num_active_by_node = [len(node.key) if (node.hit_count > 0) else 0 for node in nodes]
        num_stores_by_node = [len(node.key) for node in nodes]
        self.metrics.set_active_and_store(sum(num_active_by_node), sum(num_stores_by_node))



    ##### Internal Helper Functions #####

    def _match_prefix_helper(self, node: TreeNode, key: List, timestamp: float = None):
        query_len = len(key)
        if timestamp is None:
            timestamp = time.time()
        node.last_access_time = timestamp

        child_key = self.get_child_key_fn(key)

        value = []
        while len(key) > 0 and child_key in node.children.keys():
            child = node.children[child_key]
            child.last_access_time = timestamp
            prefix_len = self.key_match_fn(child.key, key)
            if prefix_len < len(child.key):
                new_node = self._split_node(child.key, child, prefix_len, last_access_time=timestamp)
                value.append(new_node.value)
                node = new_node
                node.hit_count += 1
                break
            else:
                value.append(child.value)
                node = child
                node.hit_count += 1
                key = key[prefix_len:]
                if len(key):
                    child_key = self.get_child_key_fn(key)

        # update metrics
        self.metrics.update_hits(query_len, len(value))
        return value, node

    def _split_node(self, key, child: TreeNode, split_len: int, last_access_time: float = None):
        # new_node -> child
        new_node = TreeNode(create_time=child.create_time, 
                            last_access_time=last_access_time, 
                            hit_count=child.hit_count)
        new_node.children = {self.get_child_key_fn(key[split_len:]): child}
        new_node.parent = child.parent
        new_node.lock_ref = child.lock_ref
        new_node.key = child.key[:split_len]
        new_node.value = child.value[:split_len]
        child.parent = new_node
        child.key = child.key[split_len:]
        child.value = child.value[split_len:]
        new_node.parent.children[self.get_child_key_fn(key)] = new_node
        return new_node

    def _insert_helper(self, node: TreeNode, key: List, value, timestamp: float = None):
        if timestamp is None:
            timestamp = time.time()
        node.last_access_time = timestamp 
        if len(key) == 0:
            return 0

        child_key = self.get_child_key_fn(key)

        total_prefix_length = 0
        while len(key) > 0 and child_key in node.children.keys():
            node = node.children[child_key]
            node.last_access_time = timestamp
            prefix_len = self.key_match_fn(node.key, key)
            total_prefix_length += prefix_len
            key = key[prefix_len:]
            value = value[prefix_len:]

            if prefix_len < len(node.key):
                new_node = self._split_node(node.key, node, prefix_len, last_access_time=timestamp)
                node = new_node

            if len(key):
                child_key = self.get_child_key_fn(key)

        if len(key):
            new_node = TreeNode(create_time=timestamp, last_access_time=timestamp)
            new_node.parent = node
            new_node.key = key
            new_node.value = value
            node.children[child_key] = new_node
            self.evictable_size_ += len(value)
        
        # update metrics
        self.metrics.update_stores(num_stores=len(key), timestamp=timestamp)
        return total_prefix_length

    def _print_helper(self, node: TreeNode, indent: int):
        """Prints the radix tree in a human-readable format."""
        stack = [(node, indent)]
        while stack:
            current_node, current_indent = stack.pop()
            print(
                " " * current_indent,
                len(current_node.key),
                current_node.key[:10],
                f"r={current_node.lock_ref}",
                current_node.value[:10],
                "hits={}".format(current_node.hit_count),
                "ct={:0.3f}, lat={:0.3}".format(current_node.create_time, 
                                               current_node.last_access_time),
            )
            for key, child in current_node.children.items():
                stack.append((child, current_indent + 2))

                assert key == self.get_child_key_fn(
                    child.key
                ), f"{key=}, {self.get_child_key_fn(child.key)=}"

    def _delete_leaf(self, node):
        for k, v in node.parent.children.items():
            if v == node:
                break
        del node.parent.children[k]
        self.evictable_size_ -= len(node.key)

    def _total_size_helper(self):
        total_size = 0
        stack = [self.root_node]
        while stack:
            current_node = stack.pop()
            total_size += len(current_node.value)
            for child in current_node.children.values():
                if child.evicted:
                    continue
                stack.append(child)
        return total_size

    def _collect_leaves(self):
        ret_list = []
        stack = [self.root_node]

        while stack:
            cur_node = stack.pop()
            if len(cur_node.children) == 0:
                ret_list.append(cur_node)
            else:
                stack.extend(cur_node.children.values())

        return ret_list
    
    def _collect_nodes(self):
        ret_list = []
        stack = [self.root_node]

        while stack:
            cur_node = stack.pop()
            if len(cur_node.children) > 0:
                ret_list.extend(cur_node.children.values())
                stack.extend(cur_node.children.values())
        return ret_list


if __name__ == "__main__":
    
    def test_string_keys(page_size=1):
        tree = RadixCache(page_size=page_size, disable=False)
        tree.insert("Hello")
        tree.pretty_print()
        tree.insert("Hello")
        tree.pretty_print()
        tree.insert("Hello_L.A.!")
        tree.pretty_print()
        tree.insert("Hello_world! Happy")
        tree.pretty_print()
        tree.insert("Hello_Mylove")
        tree.pretty_print()
        tree.insert("I love you!")
        tree.pretty_print()
        print(tree.match_prefix("I love you! aha"))
    
    # test tensor
    def test_int_keys(page_size=1):
        tree = RadixCache(page_size=page_size, disable=False)
        tree.insert([1, 2, 3, 4, 5]) 
        time.sleep(1)
        tree.insert([1, 2, 6, 7, 8]) 
        time.sleep(1)
        tree.insert([1, 2, 3, 4, 5, 9, 10])
        tree.insert([1, 2, 3, 4, 11, 12])
        time.sleep(1)
        tree.pretty_print()
        print(tree.match_prefix([1, 2, 3, 4, 5, 9]))
        # test evict
        tree.evict(5)
        tree.pretty_print()

    def test_int_key_with_vals(page_size=1):
        tree = RadixCache(page_size=page_size, disable=False)
        tree.insert([1, 2, 3, 4, 5], torch.tensor([11, 22, 33, 44, 55]))
        time.sleep(1)
        tree.insert([1, 2, 6, 7, 8], torch.tensor([11, 22, 66, 77, 88]))
        time.sleep(1)
        tree.insert([1, 2, 3, 4, 5, 9, 10], torch.tensor([11, 22, 33, 44, 55, 99, 1010]))
        tree.insert([1, 2, 3, 4, 11, 12], torch.tensor([11, 22, 33, 44, 55, 99, 1010]))
        tree.pretty_print()
        print(tree.match_prefix([1, 2, 3, 4, 5, 9]))
        # test evict
        tree.evict(5)
        tree.pretty_print()

    test_string_keys(page_size=1)
    test_string_keys(page_size=4)

    test_int_keys(page_size=1)
    test_int_keys(page_size=3)

    test_int_key_with_vals(page_size=1)
    test_int_key_with_vals(page_size=3)