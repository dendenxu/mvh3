"""Sample hardware SM activity without putting NVML calls on the training thread."""

import math
import time
from threading import Lock, Event, Thread


class GpuMetrics:
    """Keep a recent NVML GPM interval for one CUDA device, identified by UUID.

    GPU busy time includes waiting communication kernels. GPM instead measures
    active SMs, tensor cores and memory bandwidth over the sampling interval.
    Missing, failed or stale samples must never be reported as zero activity.
    """

    metrics = {2: "sm_active_percent", 5: "tensor_active_percent", 10: "dram_active_percent"}

    def __init__(self, uuid, interval=2.0):
        self.uuid, self.interval = str(uuid), float(interval)
        if not self.uuid.startswith(("GPU-", "MIG-")):
            self.uuid = "GPU-" + self.uuid
        if self.interval <= 0:
            raise ValueError("GPU sampling interval must be positive")
        self.lock, self.stopped = Lock(), Event()
        self.values, self.sampled_at, self.status = {}, 0.0, "initializing"
        self.thread = Thread(target=self.collect, name="gpu-metrics", daemon=True)
        self.thread.start()

    def publish(self, values, status):
        with self.lock:
            self.values, self.status = values, status
            self.sampled_at = time.monotonic()

    def snapshot(self):
        """Read cached values only; a stalled driver cannot block an update."""
        with self.lock:
            if self.values and time.monotonic() - self.sampled_at > 3 * self.interval:
                return {}, "stale"
            return dict(self.values), self.status

    def collect(self):
        samples, initialized = [], False
        try:
            import pynvml

            pynvml.nvmlInit()
            initialized = True
            device = pynvml.nvmlDeviceGetHandleByUUID(self.uuid)
            if not pynvml.nvmlGpmQueryDeviceSupport(device).isSupportedDevice:
                self.publish({}, "unsupported")
                return
            for _ in range(2):
                samples.append(pynvml.nvmlGpmSampleAlloc())
            previous, current = samples
            previous_valid = False
            while not self.stopped.is_set():
                try:
                    pynvml.nvmlGpmSampleGet(device, current)
                    if previous_valid:
                        request = pynvml.c_nvmlGpmMetricsGet_t()
                        request.version = pynvml.NVML_GPM_METRICS_GET_VERSION
                        request.numMetrics = len(self.metrics)
                        request.sample1, request.sample2 = previous, current
                        for metric, identifier in zip(request.metrics, self.metrics):
                            metric.metricId = identifier
                        pynvml.nvmlGpmMetricsGet(request)
                        values = {}
                        for metric, name in zip(request.metrics, self.metrics.values()):
                            if metric.nvmlReturn != 0:
                                raise pynvml.NVMLError(metric.nvmlReturn)
                            if not math.isfinite(metric.value) or not 0 <= metric.value <= 100:
                                raise ValueError("Invalid GPU activity percentage")
                            values[name] = metric.value
                        self.publish(values, "available")
                    previous, current = current, previous
                    previous_valid = True
                except Exception as error:
                    # A timeout invalidates this interval. Start a fresh pair
                    # on the next poll, without interrupting training.
                    self.publish({}, type(error).__name__)
                    previous_valid = False
                self.stopped.wait(self.interval)
        except Exception as error:
            self.publish({}, type(error).__name__)
        finally:
            for sample in samples:
                try:
                    pynvml.nvmlGpmSampleFree(sample)
                except Exception:
                    pass
            if initialized:
                try:
                    pynvml.nvmlShutdown()
                except Exception:
                    pass

    def close(self):
        self.stopped.set()

        # Driver calls can time out slowly. The daemon cleans up when they
        # return; finishing a run must not wait indefinitely for telemetry.
        self.thread.join(timeout=0.1)
