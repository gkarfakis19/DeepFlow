import sys
from tile import TiledGEMM


class GEMMCalc:
    def __init__(self, core, mem_hierarchy, dtype_bytes, flashattn_enable):
        self.core = core
        self.mem_hierarchy = mem_hierarchy
        self.dtype_bytes = dtype_bytes
        self.flashattn_enable = flashattn_enable

    def run(self, M : int, K : int, N : int, name = "", throughput: float | None = None):
        best_result = self._search_tile_space(M, K, N, throughput=throughput)
        return best_result
    
    def _search_tile_space(self, M : int, K : int, N : int, throughput: float | None = None):
        best_gemm = None
        best_time = float("inf")

        for gemm in TiledGEMM.enumerate_candidates(
            self.core,
            self.mem_hierarchy,
            M, K, N,
            self.dtype_bytes
        ):
            time = self._compute_time(gemm, throughput=throughput)
            if time < best_time:
                best_time = time
                best_gemm = gemm

        return best_gemm, best_time

    def _compute_time(self, gemm : TiledGEMM, throughput: float | None = None):
        times,_,_ = self._roofline(
            gemm.GEMM_flop,
            gemm.mem_accesses.totals(),
            gemm.bundle_util,
            flashattn_enable=self.flashattn_enable,
            throughput=throughput
        )
        return max(times)
    
    def _roofline(self, total_flops, mem_access : tuple, util : float,
                   flashattn_enable : bool, throughput: float | None = None):

        if not throughput:
            throughput = self.core.getThroughput()
        throughput *= util
        num_level = len(mem_access)
        try:
            if not flashattn_enable:
                assert mem_access[num_level - 1] > 0, f"mem_access: {mem_access}"
        except Exception as e:
            print(
                "Number of accesses to the last level of memory hierarchy cannot be zero:\n {}".format(e),
                flush=True,
            )
            sys.exit(0)
        levels_to_compute = [(i, mem_access[i]) for i in range(num_level)]

        # Compute roofline time for each level
        times = []
        inf_pts = []
        comp_ints = []
        for level_idx, num_mem in levels_to_compute:
            mem_bw = self.mem_hierarchy[level_idx].getThroughput()
            mem_latency = self.mem_hierarchy[level_idx].getLatency()
            inflection_point = float("inf") if mem_bw == 0 else throughput / mem_bw
            # print(f"Level {level_idx}: mem_bw={mem_bw}, mem_latency={mem_latency}, inflection_point={inflection_point}", flush=True)
            comp_int = float("inf") if num_mem == 0 else total_flops / num_mem
            if comp_int < inflection_point:  # mem-bound
                level_time = (float("inf") if (mem_bw == 0 or num_mem == 0) else (num_mem / mem_bw)) + mem_latency
            else:  # compute-bound
                level_time = float("inf") if (throughput == 0) else (total_flops / throughput)
            times.append(level_time)
            inf_pts.append(inflection_point)
            comp_ints.append(comp_int)
        # print("Roofline: exited {}".format(name))
        return times, inf_pts, comp_ints