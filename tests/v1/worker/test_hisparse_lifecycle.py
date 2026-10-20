# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU fault tests for ownership until DMA drain and host unregister succeed.

The boundary is release_pinned_state's success result and worker shutdown.
Weakrefs catch a missing allocation owner even after callers and tracebacks go
away. Streams and registration are injected; no accelerator is initialized.
"""

import contextlib
import gc
import logging
import mmap
import weakref
from collections import deque
from types import SimpleNamespace

import pytest
import torch

from vllm.distributed.kv_transfer.kv_connector.v1.hisparse import (
    worker as worker_module,
)
from vllm.v1.hisparse import binding, runtime
from vllm.v1.kv_offload.cpu.shared_offload_region import SharedOffloadRegion


class Owner:
    def __init__(self, **attributes):
        self.__dict__.update(attributes)


class Backend:
    def __init__(self):
        self.sync_error = None
        self.unpin_error_address = None
        self.unpins = []
        self.submit_error = None

    def synchronize(self):
        if self.sync_error is not None:
            raise self.sync_error

    def unpin_tensor(self, tensor):
        address = tensor.data_ptr()
        self.unpins.append(address)
        if address == self.unpin_error_address:
            raise RuntimeError("unregister failed")
        if hasattr(tensor, "_hisparse_registration"):
            tensor._hisparse_registration.closed = True
            del tensor._hisparse_registration

    def current_stream(self):
        return None

    def create_event(self):
        return Owner(record=lambda *args: None)

    def stream_context(self, stream):
        return contextlib.nullcontext()

    def swap_blocks_batch(self, *args):
        if self.submit_error is not None:
            raise self.submit_error

    def validate_kv_cache_config(self, config):
        pass


class Stream:
    error = None

    def wait_stream(self, other):
        pass

    def synchronize(self):
        if self.error is not None:
            raise self.error


class Region:
    def __init__(self):
        self.base_tensor = torch.empty(32, dtype=torch.int8)
        self.pinned_addresses = [self.base_tensor.data_ptr() + i for i in (0, 8, 16)]
        self.is_pinned = True
        self.cleanup_calls = 0
        self.cleanup_error = None

    def cleanup(self):
        assert not self.is_pinned
        assert not self.pinned_addresses
        self.cleanup_calls += 1
        if self.cleanup_error is not None:
            raise self.cleanup_error
        self.base_tensor = None


@pytest.fixture
def backend(monkeypatch):
    backend = Backend()
    monkeypatch.setattr(runtime, "_retained_pinned_states", {})
    monkeypatch.setattr(runtime, "_host_pools", {})
    monkeypatch.setattr(worker_module, "_retained_dma_workers", {})
    # Captured LogRecords retain exception frames and would mask ownership bugs.
    logger = logging.Logger("hisparse.lifecycle.weakref")
    logger.addHandler(logging.NullHandler())
    monkeypatch.setattr(runtime, "logger", logger)
    for module in (runtime, binding, worker_module):
        monkeypatch.setattr(module, "get_hisparse_backend", lambda: backend)
    yield backend
    assert not runtime._retained_pinned_states
    assert not worker_module._retained_dma_workers


def mapped_runtime():
    tensor = torch.empty(32, dtype=torch.int8)
    mapping = Owner(tensor=tensor, closed=False)
    tensor._hisparse_registration = mapping
    state = Owner(
        _host_cache=tensor,
        registered_host_pool=tensor,
        cache_ops=Owner(host_mapping=mapping),
        hot_backing=torch.empty(32),
    )
    return state, tensor, mapping


@pytest.mark.parametrize("phase", ["sync", "unregister"])
def test_failed_release_keeps_actual_owners_until_successful_retry(
    backend, phase, caplog
):
    state, tensor, mapping = mapped_runtime()
    refs = [weakref.ref(value) for value in (state, tensor, mapping)]
    if phase == "sync":
        backend.sync_error = RuntimeError("drain failed")
    else:
        backend.unpin_error_address = tensor.data_ptr()
    assert runtime.release_pinned_state([state], [tensor]) is False
    assert runtime.release_pinned_state([state], [tensor]) is False
    assert len(runtime._retained_pinned_states) == 1
    del state, tensor, mapping
    # Exceptions themselves hold tracebacks; remove that independent owner.
    backend.sync_error = None
    caplog.clear()
    gc.collect()
    assert all(ref() is not None for ref in refs)
    assert not refs[2]().closed
    backend.unpin_error_address = None
    assert runtime.release_pinned_state([refs[0]()], [refs[1]()]) is True
    assert runtime.release_pinned_state([], []) is True
    gc.collect()
    assert all(ref() is None for ref in refs)


def test_private_partial_release_never_repeats_successful_unregister(backend):
    first, second = torch.empty(8), torch.empty(8)
    addresses = [first.data_ptr(), second.data_ptr()]
    backend.unpin_error_address = addresses[0]
    pools = [first, second]
    assert runtime.release_pinned_state([], pools) is False
    assert len(pools) == 1 and pools[0] is first
    assert backend.unpins == [addresses[1], addresses[0]]
    backend.unpin_error_address = None
    # A fresh caller list may still contain the already released second pool.
    assert runtime.release_pinned_state([], [first, second]) is True
    assert backend.unpins == [addresses[1], addresses[0], addresses[0]]


def test_shared_partial_release_commits_each_chunk_and_preserves_region(
    backend, caplog
):
    region = Region()
    addresses = list(region.pinned_addresses)
    region_ref, tensor_ref = weakref.ref(region), weakref.ref(region.base_tensor)
    backend.unpin_error_address = addresses[1]
    assert runtime.release_pinned_state([], [], region) is False
    assert region.pinned_addresses == addresses[:2]
    assert region.is_pinned and region.cleanup_calls == 0
    del region
    caplog.clear()
    gc.collect()
    assert region_ref() is not None and tensor_ref() is not None
    backend.unpin_error_address = None
    assert runtime.release_pinned_state([], [], region_ref()) is True
    assert backend.unpins == [addresses[2], addresses[1], addresses[1], addresses[0]]
    gc.collect()
    assert region_ref() is None and tensor_ref() is None


def test_shared_sync_failure_does_not_change_registration_state(backend):
    region = Region()
    addresses = list(region.pinned_addresses)
    backend.sync_error = RuntimeError("drain failed")
    assert runtime.release_pinned_state([], [], region) is False
    assert region.pinned_addresses == addresses and region.is_pinned
    assert not backend.unpins and not region.cleanup_calls
    backend.sync_error = None
    assert runtime.release_pinned_state([], [], region) is True


def test_shared_cleanup_retry_does_not_unregister_again(backend):
    region = Region()
    addresses = list(region.pinned_addresses)
    region.cleanup_error = RuntimeError("mmap cleanup failed")
    assert runtime.release_pinned_state([], [], region) is False
    assert not region.is_pinned and not region.pinned_addresses
    assert region.base_tensor is not None
    region.cleanup_error = None
    assert runtime.release_pinned_state([], [], region) is True
    assert runtime.release_pinned_state([], [], region) is True
    assert backend.unpins == list(reversed(addresses))


@pytest.mark.parametrize("external_export", [False, True])
def test_real_shared_mmap_closes_after_owned_and_external_views_release(
    backend, external_export
):
    """Use real mmap/torch exports and the production SharedOffloadRegion cleanup."""
    mapping = mmap.mmap(-1, mmap.PAGESIZE)
    region = object.__new__(SharedOffloadRegion)
    region.mmap_obj = mapping
    region._base = torch.frombuffer(memoryview(mapping), dtype=torch.int8)
    region._views = [region._base[:8]]
    region.pinned_addresses = [region._base.data_ptr()]
    region.is_pinned = True
    region.fd = None
    region._creator = False
    region.mmap_path = "hisparse-cpu-lifecycle-test"
    config = SimpleNamespace(instance_id="real-mmap")
    pool = runtime.get_hisparse_host_pool(config, None)
    pool.registered = region.base_tensor
    pool.backing = region.base_tensor[:]
    pool.shared_region = region
    state = Owner(
        _host_cache=region.base_tensor.view(1, -1),
        registered_host_pool=region.base_tensor,
        cache_ops=object(),
        hot_backing=object(),
    )
    external = region.base_tensor[8:16] if external_export else None
    assert runtime.release_pinned_state([state], [], region) is (not external_export)
    assert pool.registered is None and pool.backing is None
    assert not hasattr(state, "_host_cache")
    assert not region.is_pinned
    if external_export:
        assert not mapping.closed and region.mmap_obj is mapping
        del external
        gc.collect()
        assert runtime.release_pinned_state([state], [], region) is True
    assert mapping.closed and region.mmap_obj is None
    assert len(backend.unpins) == 1
    runtime.drop_hisparse_host_pool(config)


def test_static_view_rollback_keeps_pool_and_original_exception(monkeypatch, backend):
    state, tensor, mapping = mapped_runtime()
    config = SimpleNamespace(
        instance_id="failed-view",
        cache_config=SimpleNamespace(get_resolved_kv_cache_layout=lambda: None),
    )
    pool = runtime.get_hisparse_host_pool(config, None)
    pool.registered = pool.backing = tensor
    context = {"layer": Owner(hisparse_cache=Owner(runtime=state, view=object()))}
    init_error = ValueError("static view failed")

    def invalid_view(*args):
        raise init_error

    backend.prepare_kv_cache_views = invalid_view
    backend.unpin_error_address = tensor.data_ptr()
    monkeypatch.setattr(binding, "allocate_hisparse_kv_caches", lambda *args: {})
    with pytest.raises(ValueError) as caught:
        binding.init_hisparse_kv_cache(None, None, [], config, context, None)
    assert caught.value is init_error
    assert runtime._host_pools[config.instance_id] is pool
    assert state.cache_ops.host_mapping is mapping and not mapping.closed
    with pytest.raises(RuntimeError, match="awaits successful teardown"):
        runtime.get_hisparse_host_pool(config, None)
    with pytest.raises(RuntimeError, match="unreleased"):
        runtime.drop_hisparse_host_pool(config)
    backend.unpin_error_address = None
    assert runtime.release_pinned_state([state], [tensor]) is True
    binding.release_hisparse_profiling_cache(context)
    binding.release_hisparse_profiling_cache(context)
    assert context["layer"].hisparse_cache.view is None
    runtime.drop_hisparse_host_pool(config)


def test_empty_profiling_context_does_not_release(monkeypatch):
    def unexpected_release(*args):
        pytest.fail("An empty profiling context has no state to release")

    monkeypatch.setattr(binding, "release_pinned_state", unexpected_release)
    binding.release_hisparse_profiling_cache({})
    binding.release_hisparse_profiling_cache({"layer": Owner(hisparse_cache=None)})


def test_static_view_error_without_handles_releases_pool_once(monkeypatch, backend):
    tensor = torch.empty(32, dtype=torch.int8)
    config = SimpleNamespace(
        instance_id="failed-static-view",
        cache_config=SimpleNamespace(get_resolved_kv_cache_layout=lambda: None),
    )
    pool = runtime.get_hisparse_host_pool(config, None)
    pool.registered = pool.backing = tensor
    error = ValueError("invalid indexer stride")

    def invalid_views(*args):
        raise error

    releases = []

    def release(*args):
        releases.append(args)
        return runtime.release_pinned_state(*args)

    backend.prepare_kv_cache_views = invalid_views
    monkeypatch.setattr(binding, "allocate_hisparse_kv_caches", lambda *args: {})
    monkeypatch.setattr(binding, "release_pinned_state", release)
    with pytest.raises(ValueError) as caught:
        binding.init_hisparse_kv_cache(None, None, [], config, {}, None)
    assert caught.value is error
    assert len(releases) == 1
    assert len(releases[0]) == 3 and releases[0][2] is None
    assert config.instance_id not in runtime._host_pools
    assert backend.unpins == [tensor.data_ptr()]


def test_partial_bind_rollback_releases_views_after_success(monkeypatch, backend):
    state, tensor, mapping = mapped_runtime()
    config = SimpleNamespace(
        instance_id="failed-buffers",
        cache_config=SimpleNamespace(get_resolved_kv_cache_layout=lambda: None),
        scheduler_config=SimpleNamespace(max_num_seqs=1),
    )
    pool = runtime.get_hisparse_host_pool(config, None)
    pool.registered = pool.backing = tensor
    handle = Owner(runtime=state, view=object())
    context = {"layer": Owner(hisparse_cache=handle)}
    backend.prepare_kv_cache_views = lambda *args: None
    init_error = ValueError("runtime buffers failed")

    def invalid_buffers(*args, **kwargs):
        raise init_error

    monkeypatch.setattr(binding, "allocate_hisparse_kv_caches", lambda *args: {})
    monkeypatch.setattr(binding, "bind_hisparse_kv_caches", lambda **kwargs: [handle])
    monkeypatch.setattr(binding, "initialize_hisparse_runtime_buffers", invalid_buffers)
    with pytest.raises(ValueError) as caught:
        binding.init_hisparse_kv_cache(None, None, [], config, context, None)
    assert caught.value is init_error
    assert config.instance_id not in runtime._host_pools
    assert handle.view is None and mapping.closed
    assert not hasattr(state, "registered_host_pool")
    assert len(backend.unpins) == 1


def test_connector_registration_failure_preserves_pool_and_original_error(
    monkeypatch, backend
):
    state, tensor, mapping = mapped_runtime()
    state.shared_host_region = None
    handle = Owner(runtime=state, view=object())
    config = SimpleNamespace(
        instance_id="failed-worker",
        compilation_config=SimpleNamespace(
            static_forward_context={"layer": Owner(hisparse_cache=handle)}
        ),
        scheduler_config=SimpleNamespace(max_num_seqs=1),
    )
    pool = runtime.get_hisparse_host_pool(config, None)
    pool.registered = pool.backing = tensor
    hot_spec = type("HotSpec", (), {})
    monkeypatch.setattr(worker_module, "HiSparseHotSpec", hot_spec)
    monkeypatch.setattr(worker_module, "_is_hisparse_host_writer", lambda region: True)
    cache_config = SimpleNamespace(
        kv_cache_groups=[
            SimpleNamespace(
                kv_cache_spec=hot_spec(),
                layer_names=["layer" + worker_module.HISPARSE_HOT_SUFFIX],
            )
        ],
        hisparse_host_num_blocks=1,
    )
    worker = worker_module.HiSparseConnectorWorker(config, cache_config)
    init_error = ValueError("worker initialization failed")

    def invalid_initialize(*args, **kwargs):
        raise init_error

    worker.initialize = invalid_initialize
    backend.unpin_error_address = tensor.data_ptr()
    with pytest.raises(ValueError) as caught:
        worker.register_kv_caches({})
    assert caught.value is init_error
    assert runtime._host_pools[config.instance_id] is pool
    assert state.cache_ops.host_mapping is mapping and not mapping.closed
    backend.unpin_error_address = None
    assert runtime.release_pinned_state([state], [tensor]) is True
    runtime.drop_hisparse_host_pool(config)


@pytest.mark.parametrize("shared", [False, True])
def test_allocation_error_preserves_prior_registrations(monkeypatch, backend, shared):
    tensor = torch.empty(32, dtype=torch.int8)
    region = Region()
    region.pinned_addresses.clear()
    region.is_pinned = False
    init_error = ValueError("registration failed")
    calls = 0

    def allocate_or_pin(*args):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise init_error
        return tensor, tensor

    monkeypatch.setattr(runtime, "check_hisparse_host_memory", lambda size: None)
    monkeypatch.setattr(runtime, "allocate_pinned_host_pool", allocate_or_pin)
    monkeypatch.setattr(runtime, "pin_tensor", allocate_or_pin)
    monkeypatch.setattr(runtime, "SharedOffloadRegion", lambda **kwargs: region)
    monkeypatch.setattr(
        runtime,
        "get_tp_group",
        lambda: SimpleNamespace(barrier=lambda: None, rank_in_group=0, world_size=1),
    )
    monkeypatch.setattr(
        runtime, "_hisparse_registration_ranges", lambda *args: [(0, 8), (8, 16)]
    )
    backend.sync_error = RuntimeError("drain failed")
    config = SimpleNamespace(
        instance_id="allocation", parallel_config=SimpleNamespace(data_parallel_index=0)
    )
    with pytest.raises(ValueError) as caught:
        runtime.allocate_hisparse_host_pools(
            config, [8, 8], 1, 32, use_shared_host_pool=shared
        )
    assert caught.value is init_error
    assert len(runtime._retained_pinned_states) == 1
    backend.sync_error = None
    assert (
        runtime.release_pinned_state(
            [], [] if shared else [tensor], region if shared else None
        )
        is True
    )


def worker_with_cache(backend):
    worker = worker_module.HiSparseConnectorWorker(
        SimpleNamespace(instance_id="worker"), None
    )
    worker._initialized = True
    worker._slot_mapping_staging = None
    worker.dma_stream = Stream()
    worker.host_write_event = backend.create_event()
    worker._pending_dma_descriptors = deque()
    worker._dma_free_descriptors = []
    worker._pending_transfer_events = deque()
    worker._enqueued_transfer_ids = []
    state, tensor, _ = mapped_runtime()
    worker.cache_handles = [Owner(runtime=state)]
    worker.pinned_host_pools = [tensor]
    worker.shared_host_region = None
    return worker


def test_failed_dma_and_drain_keep_descriptors_and_original_error(backend):
    worker = worker_with_cache(backend)
    tensors = [torch.zeros(1, dtype=torch.int64) for _ in range(3)]
    descriptors = worker_module._DMADescriptors(
        *tensors, *(tensor.numpy() for tensor in tensors)
    )
    del tensors
    descriptor_ref, worker_ref = weakref.ref(descriptors.src), weakref.ref(worker)
    pool_ref = weakref.ref(worker.pinned_host_pools[0])
    submit_error, drain_error = (
        RuntimeError("partial DMA"),
        RuntimeError("drain failed"),
    )
    backend.submit_error = submit_error
    worker.dma_stream.error = drain_error
    with pytest.raises(RuntimeError) as caught:
        worker._submit_dma_descriptors(descriptors, 1, transfer_ids=(7,))
    assert caught.value is submit_error and caught.value.__cause__ is drain_error
    assert not worker._pending_dma_descriptors and not worker._dma_free_descriptors
    assert not worker._pending_transfer_events and not worker._enqueued_transfer_ids
    del caught, submit_error, drain_error, descriptors, worker
    backend.submit_error = None
    worker_ref().dma_stream.error = None
    gc.collect()
    assert descriptor_ref() is not None and pool_ref() is not None
    worker_ref().shutdown()
    gc.collect()
    assert descriptor_ref() is None and pool_ref() is None and worker_ref() is None


def test_failed_dma_with_successful_drain_can_release_descriptors(backend):
    worker = worker_with_cache(backend)
    tensors = [torch.zeros(1, dtype=torch.int64) for _ in range(3)]
    descriptors = worker_module._DMADescriptors(
        *tensors, *(tensor.numpy() for tensor in tensors)
    )
    descriptor_ref = weakref.ref(descriptors.src)
    del tensors
    backend.submit_error = ValueError("submission failed")
    with pytest.raises(ValueError, match="submission failed"):
        worker._submit_dma_descriptors(descriptors, 1)
    backend.submit_error = None
    del descriptors
    gc.collect()
    assert descriptor_ref() is None
    assert not worker._failed_dma_descriptors
    assert not worker_module._retained_dma_workers
    worker.shutdown()


@pytest.mark.parametrize("phase", ["stream", "unregister"])
def test_shutdown_failure_retains_worker_and_retry_is_idempotent(
    backend, phase, caplog
):
    worker = worker_with_cache(backend)
    worker_ref = weakref.ref(worker)
    pool_ref = weakref.ref(worker.pinned_host_pools[0])
    if phase == "stream":
        worker.dma_stream.error = RuntimeError("stream drain failed")
        with pytest.raises(RuntimeError, match="stream drain failed"):
            worker.shutdown()
        assert not backend.unpins
    else:
        backend.unpin_error_address = worker.pinned_host_pools[0].data_ptr()
        worker.shutdown()
    assert worker._initialized
    worker.dma_stream.error = None
    del worker
    caplog.clear()
    gc.collect()
    assert worker_ref() is not None and pool_ref() is not None
    backend.unpin_error_address = None
    worker = worker_ref()
    worker.shutdown()
    worker.shutdown()
    assert not worker._initialized
    del worker
    gc.collect()
    assert worker_ref() is None and pool_ref() is None
