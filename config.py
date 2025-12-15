from dataclasses import dataclass, field
import math
from collections import namedtuple as _namedtuple
from typing import Dict, List, Optional, Sequence, Tuple

import yaml as _yaml
from yaml import YAMLError as _YAMLError


_PRECISION_DTYPE_BYTES = {
    "fp8": 1.0,
    "fp16": 2.0,
    "half": 2.0,
    "bf16": 2.0,
    "fp32": 4.0,
    "single": 4.0,
}


@dataclass(frozen=True)
class PrecisionConfig:
    tensor: float
    mixed_precision: bool
    param_storage_mode: str
    kv_cache: float
    parameters: float
    gradients: float
    grad_communication: float
    optimizer_states: float
    stats: float
    master_parameters: float

    @property
    def activations(self) -> float:
        return self.tensor

    @property
    def tensor_format(self) -> float:
        return self.tensor

    @property
    def requires_master_copy(self) -> bool:
        return self.master_parameters > 0.0

def _coerce_precision_value(value, *, tensor_bytes: Optional[float] = None, allow_as_tensor: bool = False) -> float:
    if isinstance(value, (int, float)):
        if value <= 0:
            raise ValueError("precision byte size must be positive")
        return float(value)

    if not isinstance(value, str):
        raise TypeError(f"Unsupported precision specification type: {type(value)!r}")

    normalized = value.strip().lower()
    if allow_as_tensor and normalized == "as_tensor_format":
        return float(tensor_bytes)

    if normalized in _PRECISION_DTYPE_BYTES:
        return _PRECISION_DTYPE_BYTES[normalized]

    parsed = float(normalized)
    if parsed <= 0:
        raise ValueError("precision byte size must be positive")
    return parsed


def _parse_precision_block(spec: dict) -> PrecisionConfig:
    tensor_bytes = _coerce_precision_value(spec["tensor_format"])
    mixed = bool(spec["mixed_precision"])

    raw_mode = str(spec["param_storage_mode"]).strip().lower()
    if raw_mode not in {"as_tensor_format", "tensor_plus_fp32_master", "fp32_params"}:
        raise ValueError(
            "sw_param.precision.param_storage_mode must be one of 'as_tensor_format', "
            "'tensor_plus_fp32_master', or 'fp32_params'"
        )
    if raw_mode == "tensor_plus_fp32_master" and not mixed:
        raise ValueError("tensor_plus_fp32_master requires mixed_precision=true")

    kv_cache_bytes = _coerce_precision_value(
        spec["kv_cache"],
        tensor_bytes=tensor_bytes,
        allow_as_tensor=True,
    )

    if not mixed:
        parameter_bytes = tensor_bytes
        gradient_bytes = tensor_bytes
        grad_comm_bytes = tensor_bytes
        optimizer_bytes = tensor_bytes
        stats_bytes = tensor_bytes
        effective_mode = "as_tensor_format"
        master_bytes = 0.0
    else:
        effective_mode = raw_mode
        optimizer_bytes = 4.0 # FP32
        stats_bytes = 4.0 # FP32
        master_bytes = 4.0 if raw_mode == "tensor_plus_fp32_master" else 0.0
        if raw_mode == "fp32_params":
            parameter_bytes = 4.0 # FP32
            gradient_bytes = 4.0 # FP32
            grad_comm_bytes = 4.0 # FP32
        else:
            parameter_bytes = tensor_bytes
            gradient_bytes = tensor_bytes
            grad_comm_bytes = tensor_bytes

    return PrecisionConfig(
        tensor=tensor_bytes,
        mixed_precision=mixed,
        param_storage_mode=effective_mode,
        kv_cache=kv_cache_bytes,
        parameters=parameter_bytes,
        gradients=gradient_bytes,
        grad_communication=grad_comm_bytes,
        optimizer_states=optimizer_bytes,
        stats=stats_bytes,
        master_parameters=master_bytes,
    )


@dataclass
class CoreConfig:
    nominal_power_per_mcu: float
    nominal_flop_rate_per_mcu: float
    nominal_energy_per_flop: float
    nominal_voltage: float
    threshold_voltage: float
    margin_voltage: float
    operating_area_per_mcu: float
    num_mcu_per_bundle: int
    FMA_dims: tuple
    dataflow: str
    util: float
    num_bundles: int = None
    operating_frequency: float = None
    nominal_frequency: float = None
    nominal_area_per_mcu: float = None

    @classmethod
    def from_dict(cls, core_config_dict):
        return cls(
            nominal_power_per_mcu=core_config_dict.get("nominal_power_per_mcu", 0.1),
            nominal_flop_rate_per_mcu=core_config_dict["nominal_flop_rate_per_mcu"],
            nominal_energy_per_flop=core_config_dict["nominal_energy_per_flop"],
            nominal_voltage=core_config_dict.get("nominal_voltage", 0.1),
            threshold_voltage=core_config_dict.get("threshold_voltage", 0.1),
            margin_voltage=core_config_dict.get("margin_voltage", 0.1),
            operating_area_per_mcu=core_config_dict.get("operating_area_per_mcu", 0.1),
            num_mcu_per_bundle=core_config_dict["num_mcu_per_bundle"],
            FMA_dims=(core_config_dict["FMA_d1"], core_config_dict["FMA_d2"]),
            dataflow=core_config_dict["dataflow"],
            util=core_config_dict["util"],
            num_bundles=core_config_dict.get("num_bundles", None),
            operating_frequency=core_config_dict.get("operating_frequency", None),
            nominal_frequency=core_config_dict.get("nominal_frequency", None),
            nominal_area_per_mcu=core_config_dict.get("nominal_area_per_mcu", None),
        )


@dataclass
class DRAMConfig:
    dynamic_energy_per_bit: float
    static_power_per_bit: float
    area_per_bit: float
    stack_capacity: float
    area_per_stack: float
    latency: float
    mem_ctrl_area: float
    nominal_voltage: float
    threshold_voltage: float
    margin_voltage: float
    num_links_per_mm: int
    num_links_per_stack: int
    max_voltage: float
    util: float
    size: float = None
    bandwidth: float = None
    num_stacks: int = None
    operating_frequency: float = None
    nominal_frequency: float = None

    @classmethod
    def from_dict(cls, dram_config_dict):
        return cls(
            dynamic_energy_per_bit=dram_config_dict["dynamic_energy_per_bit"],
            static_power_per_bit=dram_config_dict.get("static_power_per_bit", 0.1),
            area_per_bit=dram_config_dict.get("area_per_bit", 0.1),
            stack_capacity=dram_config_dict.get("stack_capacity", 0.1),
            area_per_stack=dram_config_dict.get("area_per_stack", 0.1),
            latency=dram_config_dict["latency"],
            mem_ctrl_area=dram_config_dict.get("mem_ctrl_area", 0.1),
            nominal_voltage=dram_config_dict.get("nominal_voltage", 0.1),
            threshold_voltage=dram_config_dict.get("threshold_voltage", 0.1),
            margin_voltage=dram_config_dict.get("margin_voltage", 0.1),
            num_links_per_mm=dram_config_dict.get("num_links_per_mm", 1),
            num_links_per_stack=dram_config_dict.get("num_links_per_stack", 1),
            max_voltage=dram_config_dict.get("max_voltage", 0.1),
            util=dram_config_dict["util"],
            size=dram_config_dict.get("size", None),
            bandwidth=dram_config_dict.get("bandwidth", None),
            num_stacks=dram_config_dict.get("num_stacks", None),
            operating_frequency=dram_config_dict.get("operating_frequency", None),
            nominal_frequency=dram_config_dict.get("nominal_frequency", None),
        )


@dataclass
class SRAMConfig:
    dynamic_energy_per_bit: float
    static_power_per_bit: float
    area_per_bit: float
    bank_capacity: float
    controller_area_per_link: float
    latency: float
    overhead: float
    util: float
    size: float = None
    bandwidth: float = None

    @classmethod
    def from_dict(cls, sram_config_dict):
        return cls(
            dynamic_energy_per_bit=sram_config_dict["dynamic_energy_per_bit"],
            static_power_per_bit=sram_config_dict.get("static_power_per_bit", 0.1),
            area_per_bit=sram_config_dict.get("area_per_bit", 0.1),
            bank_capacity=sram_config_dict.get("bank_capacity", 0.1),
            controller_area_per_link=sram_config_dict.get("controller_area_per_link", 0.1),
            latency=sram_config_dict["latency"],
            overhead=sram_config_dict.get("overhead", 0.1),
            util=sram_config_dict["util"],
            size=sram_config_dict.get("size", None),
            bandwidth=sram_config_dict.get("bandwidth", None),
        )


@dataclass
class TechConfig:
    core: CoreConfig
    DRAM: DRAMConfig
    SRAML2: SRAMConfig
    SRAML1: SRAMConfig
    SRAMR: SRAMConfig

    @classmethod
    def from_dict(cls, tech_config_dict):
        return cls(
            core=CoreConfig.from_dict(tech_config_dict["core"]),
            DRAM=DRAMConfig.from_dict(tech_config_dict["DRAM"]),
            SRAML2=SRAMConfig.from_dict(tech_config_dict["SRAM-L2"]),
            SRAML1=SRAMConfig.from_dict(tech_config_dict["SRAM-L1"]),
            SRAMR=SRAMConfig.from_dict(tech_config_dict["SRAM-R"]),
        )


@dataclass
class AreaBreakdownConfig:
    proc_chip_area_budget: float
    core: float
    DRAM: float
    L2: float
    L1: float
    reg_mem: float
    node_area_budget: float
    network: "NetworkAreaConfig"

    @classmethod
    def from_dict(cls, area_config_dict):
        return cls(
            proc_chip_area_budget=area_config_dict["proc_chip_area_budget"],
            core=area_config_dict["core"],
            DRAM=area_config_dict["DRAM"],
            L2=area_config_dict["L2"],
            L1=area_config_dict["L1"],
            reg_mem=area_config_dict["reg_mem"],
            node_area_budget=area_config_dict["device_area_budget"],
            network=NetworkAreaConfig.from_dict(area_config_dict["network"]),
        )


@dataclass
class PerimeterBreakdownConfig:
    DRAM: float
    inter_node: float
    intra_node: float

    @classmethod
    def from_dict(cls, perimeter_config_dict):
        return cls(
            DRAM=perimeter_config_dict["DRAM"],
            inter_node=perimeter_config_dict["inter_node"],
            intra_node=perimeter_config_dict["intra_node"],
        )


@dataclass
class NetworkAreaConfig:
    inter_node: float
    intra_node: float

    @classmethod
    def from_dict(cls, network_config_dict):
        return cls(
            inter_node=network_config_dict["inter_node"],
            intra_node=network_config_dict["intra_node"],
        )


@dataclass
class PowerBreakdownConfig:
    TDP: float
    core: float
    DRAM: float
    L2: float
    L1: float
    reg_mem: float
    network: "NetworkPowerConfig"

    @classmethod
    def from_dict(cls, power_config_dict):
        return cls(
            TDP=power_config_dict["TDP"],
            core=power_config_dict["core"],
            DRAM=power_config_dict["DRAM"],
            L2=power_config_dict["L2"],
            L1=power_config_dict["L1"],
            reg_mem=power_config_dict["reg_mem"],
            network=NetworkPowerConfig.from_dict(power_config_dict["network"]),
        )


@dataclass
class NetworkPowerConfig:
    inter_node: float
    intra_node: float

    @classmethod
    def from_dict(cls, network_power_config_dict):
        return cls(
            inter_node=network_power_config_dict["inter_node"],
            intra_node=network_power_config_dict["intra_node"],
        )


@dataclass(frozen=True)
class NetworkDimensionLayout:
    id: str
    label: str
    size: int
    topology_type: str
    bandwidth: float
    util: float
    latency: float
    size_2d: Optional[Tuple[int, int]] = None
    collective_override: Dict[str, str] = field(default_factory=dict)
    parallelisms: Tuple[str, ...] = field(default_factory=tuple)
    energy_per_bit: float = 0.0
    optimize_2dmap: bool = False

    @classmethod
    def from_raw(
        cls,
        raw: dict,
        *,
        parallelism_params: Dict[str, object],
        index: int,
    ) -> "NetworkDimensionLayout":
        if not isinstance(raw, dict):
            raise TypeError("each network dimension must be a mapping")

        raw_id = raw.get("id")
        dim_id = str(raw_id) if raw_id is not None else f"dim{index}"
        label = str(raw.get("label", dim_id))

        if "size" not in raw:
            raise ValueError(f"network dimension '{label}' is missing required field 'size'")
        size_raw = raw["size"]
        size_mode: str
        size_value: Optional[int] = None
        size_2d: Optional[Tuple[int, int]] = None
        tuple_entries: Optional[Tuple[object, object]] = None
        if isinstance(size_raw, str):
            normalized_size = size_raw.strip()
            if normalized_size.startswith("(") and normalized_size.endswith(")"):
                tuple_entries = tuple(
                    part.strip() for part in normalized_size[1:-1].split(",")
                )  # type: ignore[assignment]
                size_mode = "tuple"
            elif normalized_size.lower() == "auto":
                size_mode = "auto_scalar"
            else:
                try:
                    size_value = int(size_raw)
                except (TypeError, ValueError) as exc:
                    raise ValueError(f"network dimension '{label}' size must be an integer or tuple") from exc
                if size_value < 1:
                    raise ValueError(f"network dimension '{label}' size must be >= 1")
                size_mode = "scalar"
        elif isinstance(size_raw, (list, tuple)):
            if len(size_raw) != 2:
                raise ValueError(f"network dimension '{label}' 2D size must have exactly two entries")
            tuple_entries = tuple(size_raw)  # type: ignore[assignment]
            size_mode = "tuple"
        else:
            size_mode = "auto_scalar"

        topo_dict = raw.get("topology")
        if not isinstance(topo_dict, dict):
            raise ValueError(f"network dimension '{label}' requires a 'topology' mapping")
        if "type" not in topo_dict:
            raise ValueError(f"network dimension '{label}' topology missing required 'type'")
        topo_type = str(topo_dict["type"])

        if "bandwidth" not in topo_dict:
            raise ValueError(f"network dimension '{label}' topology missing required 'bandwidth'")
        bandwidth = parse_bandwidth_string(topo_dict["bandwidth"])

        try:
            util = float(topo_dict.get("util", 1.0))
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"network dimension '{label}' topology util must be numeric"
            ) from exc
        if util <= 0:
            raise ValueError(f"network dimension '{label}' topology util must be > 0")

        if "energy_per_bit" not in topo_dict:
            raise ValueError(
                f"network dimension '{label}' topology missing required 'energy_per_bit'"
            )
        try:
            energy_per_bit = float(topo_dict["energy_per_bit"])
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"network dimension '{label}' energy_per_bit must be numeric"
            ) from exc
        if energy_per_bit < 0:
            raise ValueError(
                f"network dimension '{label}' energy_per_bit must be >= 0"
            )

        if "latency" not in topo_dict:
            raise ValueError(f"network dimension '{label}' topology missing required 'latency'")
        try:
            latency = float(topo_dict["latency"])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"network dimension '{label}' latency must be numeric") from exc

        raw_optimize = topo_dict.get("optimize_2dmap", False)
        if raw_optimize not in (True, False):
            raise ValueError(
                f"network dimension '{label}' topology optimize_2dmap must be a boolean when provided"
            )
        optimize_2dmap = bool(raw_optimize)
        if optimize_2dmap:
            normalized_topo = topo_type.lower()
            if normalized_topo not in {"mesh2d", "torus2d", "kingmesh2d"}:
                raise ValueError(
                    f"network dimension '{label}' sets optimize_2dmap but topology type '{topo_type}'"
                    " is not Mesh2D/Torus2D/KingMesh2D"
                )

        collectives_raw = raw.get("collective_override")
        if collectives_raw is None and "collectives" in raw:
            collectives_raw = raw.get("collectives")
        if not collectives_raw:
            collectives_raw = {}
        if not isinstance(collectives_raw, dict):
            raise ValueError(f"network dimension '{label}' collective_override must be a mapping if provided")
        collective_override = {str(k): str(v) for k, v in collectives_raw.items()}

        parallelisms_raw = raw.get("parallelisms", [])
        if parallelisms_raw is None:
            parallelisms_raw = []
        if not isinstance(parallelisms_raw, Sequence) or isinstance(parallelisms_raw, (str, bytes)):
            raise ValueError(
                f"network dimension '{label}' parallelisms must be a sequence of names"
            )

        normalized_parallelisms: List[str] = []
        alias_map: Dict[str, str] = {}
        for entry in parallelisms_raw:
            name = str(entry).strip()
            if not name:
                raise ValueError(f"network dimension '{label}' has an empty parallelism name")
            normalized = name.lower()
            normalized_parallelisms.append(normalized)
            alias_map[normalized] = name

        computed_product: Optional[int] = None
        auto_product: Optional[int] = None

        if size_mode in {"auto_scalar", "tuple"}:
            auto_product = _compute_dimension_parallelism_product(
                dimension_label=label,
                normalized_names=tuple(normalized_parallelisms),
                alias_map=alias_map,
                parallelism_params=parallelism_params,
            )
            if auto_product < 1:
                raise ValueError(f"network dimension '{label}' inferred size must be >= 1")

        if size_mode == "auto_scalar":
            size_value = auto_product
            computed_product = auto_product
        elif size_mode == "tuple":
            assert tuple_entries is not None
            resolved: List[Optional[int]] = [None, None]
            auto_entries = 0
            for idx, entry in enumerate(tuple_entries):
                if isinstance(entry, str):
                    normalized = entry.strip().lower()
                    if normalized == "auto":
                        auto_entries += 1
                        if auto_entries > 1:
                            raise ValueError(
                                f"network dimension '{label}' 2D size may include at most one 'auto' entry"
                            )
                        resolved[idx] = None
                        continue
                    if normalized not in parallelism_params:
                        alias = alias_map.get(normalized, normalized)
                        raise ValueError(
                            f"network dimension '{label}' 2D size references parallelism '{alias}' "
                            "which is not defined in parallelism"
                        )
                    try:
                        factor = int(parallelism_params[normalized])
                    except (TypeError, ValueError) as exc:
                        alias = alias_map.get(normalized, normalized)
                        raise ValueError(
                            f"network dimension '{label}' 2D size parallelism '{alias}' must be an integer"
                        ) from exc
                    if factor < 1:
                        alias = alias_map.get(normalized, normalized)
                        raise ValueError(
                            f"network dimension '{label}' 2D size parallelism '{alias}' must be >= 1"
                        )
                    resolved[idx] = factor
                else:
                    try:
                        factor = int(entry)
                    except (TypeError, ValueError) as exc:
                        raise ValueError(
                            f"network dimension '{label}' 2D size entries must be integers, parallelism names, or 'auto'"
                        ) from exc
                    if factor < 1:
                        raise ValueError(f"network dimension '{label}' 2D size entries must be >= 1")
                    resolved[idx] = factor

            known_product = 1
            for val in resolved:
                if val:
                    known_product *= val

            if auto_entries:
                if auto_product is None:
                    auto_product = _compute_dimension_parallelism_product(
                        dimension_label=label,
                        normalized_names=tuple(normalized_parallelisms),
                        alias_map=alias_map,
                        parallelism_params=parallelism_params,
                    )
                if known_product == 0:
                    raise ValueError(f"network dimension '{label}' 2D size known entries must be > 0")
                if auto_product % known_product != 0:
                    raise ValueError(
                        f"network dimension '{label}' 2D size mismatch: auto product {auto_product} "
                        f"is not divisible by provided factors {tuple_entries}"
                    )
                auto_value = auto_product // known_product
                if auto_value < 1:
                    raise ValueError(
                        f"network dimension '{label}' 2D size auto-resolved entry must be >= 1 (got {auto_value})"
                    )
                for idx in range(2):
                    if resolved[idx] is None:
                        resolved[idx] = auto_value
                        break
            else:
                if auto_product is None:
                    auto_product = _compute_dimension_parallelism_product(
                        dimension_label=label,
                        normalized_names=tuple(normalized_parallelisms),
                        alias_map=alias_map,
                        parallelism_params=parallelism_params,
                    )
                if known_product != auto_product:
                    raise ValueError(
                        f"network dimension '{label}' 2D size mismatch: provided shape {tuple_entries} "
                        f"product {known_product} does not match parallelism product {auto_product}"
                    )

            if resolved[0] is None or resolved[1] is None:
                raise ValueError(f"network dimension '{label}' 2D size could not be fully resolved")

            size_2d = (int(resolved[0]), int(resolved[1]))
            size_value = int(size_2d[0]) * int(size_2d[1])
            computed_product = auto_product
        else:
            computed_product = None

        if size_value is None:
            raise ValueError(f"network dimension '{label}' size could not be resolved")

        _validate_dimension_parallelisms(
            dimension_label=label,
            dimension_size=int(size_value) if size_value is not None else 0,
            normalized_names=tuple(normalized_parallelisms),
            alias_map=alias_map,
            parallelism_params=parallelism_params,
            expected_product=computed_product,
        )

        return cls(
            id=dim_id,
            label=label,
            size=int(size_value) if size_value is not None else 0,
            size_2d=size_2d,
            topology_type=topo_type,
            bandwidth=bandwidth,
            util=util,
            latency=latency,
            collective_override=collective_override,
            parallelisms=tuple(normalized_parallelisms),
            energy_per_bit=energy_per_bit,
            optimize_2dmap=optimize_2dmap,
        )

    @property
    def effective_bandwidth(self) -> float:
        return float(self.bandwidth) * float(self.util)


def _validate_dimension_parallelisms(
    *,
    dimension_label: str,
    dimension_size: int,
    normalized_names: Tuple[str, ...],
    alias_map: Dict[str, str],
    parallelism_params: Dict[str, object],
    expected_product: Optional[int] = None,
) -> None:
    if not normalized_names:
        return

    product = expected_product
    if product is None:
        product = _compute_dimension_parallelism_product(
            dimension_label=dimension_label,
            normalized_names=normalized_names,
            alias_map=alias_map,
            parallelism_params=parallelism_params,
        )

    if product != dimension_size:
        readable = [alias_map.get(name, name) for name in normalized_names]
        raise ValueError(
            f"Network dimension '{dimension_label}' size mismatch: declared size {dimension_size} "
            f"but parallelism factors ({readable}) imply {product}"
        )


def _compute_dimension_parallelism_product(
    *,
    dimension_label: str,
    normalized_names: Tuple[str, ...],
    alias_map: Dict[str, str],
    parallelism_params: Dict[str, object],
) -> int:
    product = 1
    for name in normalized_names:
        if name not in parallelism_params:
            alias = alias_map.get(name, name)
            raise ValueError(
                f"network dimension '{dimension_label}' references parallelism '{alias}' "
                "which is not defined in parallelism"
            )
        value = parallelism_params[name]
        if value in (None, False):
            alias = alias_map.get(name, name)
            raise ValueError(
                f"network dimension '{dimension_label}' parallelism '{alias}' must have a "
                "positive parallelism factor"
            )
        try:
            factor = int(value)
        except (TypeError, ValueError) as exc:
            alias = alias_map.get(name, name)
            raise ValueError(
                f"parallelism.{alias} must be an integer to compute network dimension sizes"
            ) from exc
        if factor < 1:
            alias = alias_map.get(name, name)
            raise ValueError(
                f"parallelism.{alias} must be >= 1 to compute network dimension sizes"
            )
        product *= factor

    return product


def _parse_network_layout(
    network_spec,
    parallelism_params: Dict[str, object],
) -> Tuple[Tuple[NetworkDimensionLayout, ...], Tuple[Tuple[int, int, float], ...], "NetworkOverlapConfig"]:
    if network_spec is None:
        raise ValueError("network section must be specified and include overlap settings")

    if not isinstance(network_spec, dict):
        raise ValueError("network must be provided as a mapping to supply overlap settings")

    faulty_links: Tuple[Tuple[int, int, float], ...] = _parse_faulty_links("network", network_spec.get("faulty_links", []))
    overlap_config = _parse_network_overlap(network_spec.get("overlap"))
    dimensions_spec = network_spec.get("dimensions")
    if dimensions_spec is None:
        raise ValueError("network.dimensions must be specified when network is a mapping")

    if not isinstance(dimensions_spec, Sequence) or isinstance(dimensions_spec, (str, bytes)):
        raise ValueError("network.dimensions must be a sequence of dimension mappings")

    dimensions: List[NetworkDimensionLayout] = []
    for index, entry in enumerate(dimensions_spec):
        dimensions.append(
            NetworkDimensionLayout.from_raw(
                entry,
                parallelism_params=parallelism_params,
                index=index,
            )
        )
    return tuple(dimensions), faulty_links, overlap_config


@dataclass(frozen=True)
class NetworkOverlapConfig:
    tp_overlap: float
    tp_sp_overlap: float
    cp_overlap: float


@dataclass(frozen=True)
class NetworkLayoutConfig:
    dimensions: Tuple[NetworkDimensionLayout, ...]
    faulty_links: Tuple[Tuple[int, int, float], ...] = field(default_factory=tuple)
    parallelism_map: Dict[str, NetworkDimensionLayout] = field(default_factory=dict)
    overlap_config: "NetworkOverlapConfig" = None

    def primary_dimension(self) -> Optional[NetworkDimensionLayout]:
        return self.dimensions[0] if self.dimensions else None

    def dimension_for_parallelism(self, name: str) -> Optional[NetworkDimensionLayout]:
        normalized = str(name).strip().lower()
        if normalized in self.parallelism_map:
            return self.parallelism_map[normalized]
        return self.primary_dimension()

    def link_for_parallelism(self, name: str) -> Tuple[float, float]:
        dim = self.dimension_for_parallelism(name)
        if dim is None:
            return 0.0, 0.0
        return dim.effective_bandwidth, dim.latency


def _parse_faulty_links(owner_label: str, faulty_links_raw) -> Tuple[Tuple[int, int, float], ...]:
    entries: List[Tuple[int, int, float]] = []
    if not faulty_links_raw:
        return tuple()
    if not isinstance(faulty_links_raw, Sequence) or isinstance(faulty_links_raw, (str, bytes)):
        raise ValueError(
            f"{owner_label} faulty_links must be a sequence of [src, dst, weight] entries"
        )
    for idx, entry in enumerate(faulty_links_raw):
        if not isinstance(entry, Sequence) or isinstance(entry, (str, bytes)) or len(entry) != 3:
            raise ValueError(
                f"{owner_label} faulty_links[{idx}] must be a three-item sequence [src, dst, weight]"
            )
        src_raw, dst_raw, weight_raw = entry
        try:
            src = int(src_raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"{owner_label} faulty_links[{idx}][0] must be an integer endpoint"
            ) from exc
        try:
            dst = int(dst_raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"{owner_label} faulty_links[{idx}][1] must be an integer endpoint"
            ) from exc
        if src < 0 or dst < 0:
            raise ValueError(
                f"{owner_label} faulty_links[{idx}] endpoints must be >= 0"
            )
        try:
            weight = float(weight_raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"{owner_label} faulty_links[{idx}][2] must be a numeric reliability weight"
            ) from exc
        if weight < 0.0 or weight > 1.0:
            raise ValueError(
                f"{owner_label} faulty_links[{idx}] weight must be between 0.0 and 1.0"
            )
        entries.append((src, dst, weight))
    return tuple(entries)

def _parse_network_overlap(overlap_raw) -> "NetworkOverlapConfig":
    if not isinstance(overlap_raw, dict):
        raise ValueError("network.overlap must be a mapping with tp_overlap, tp_sp_overlap, and cp_overlap")
    required_fields = ("tp_overlap", "tp_sp_overlap", "cp_overlap")
    values = {}
    for field in required_fields:
        if field not in overlap_raw:
            raise ValueError(f"network.overlap missing required field '{field}'")
        try:
            val = float(overlap_raw[field])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"network.overlap.{field} must be numeric") from exc
        if val < 0.0 or val > 1.0:
            raise ValueError(f"network.overlap.{field} must be between 0.0 and 1.0")
        values[field] = val
    return NetworkOverlapConfig(
        tp_overlap=values["tp_overlap"],
        tp_sp_overlap=values["tp_sp_overlap"],
        cp_overlap=values["cp_overlap"],
    )


def _build_network_layout_config(
    dimensions: Sequence[NetworkDimensionLayout],
    faulty_links: Sequence[Tuple[int, int, float]] = (),
    overlap_config: Optional["NetworkOverlapConfig"] = None,
) -> NetworkLayoutConfig:
    parallelism_map: Dict[str, NetworkDimensionLayout] = {}
    for dim in dimensions:
        for pname in dim.parallelisms:
            if pname in parallelism_map:
                raise ValueError(
                    f"parallelism '{pname}' assigned to multiple network dimensions"
                )
            parallelism_map[pname] = dim
    return NetworkLayoutConfig(
        dimensions=tuple(dimensions),
        faulty_links=tuple(faulty_links),
        parallelism_map=parallelism_map,
        overlap_config=overlap_config,
    )


_PARALLELISM_DEFAULTS: Dict[str, object] = {
    "auto": False,
    "dp": 1,
    "lp": 1,
    "mb": 1,
    "kp_hidden_dim1": 1,
    "kp_softmax_dim1": 1,
    "kp_embedding_dim1": 1,
    "kp_projection_dim1": 1,
    "kp_hidden_dim2": 1,
    "kp_softmax_dim2": 1,
    "kp_embedding_dim2": 1,
    "kp_projection_dim2": 1,
    "kp_hidden_type": -1,
    "kp_softmax_type": -1,
    "kp_embedding_type": -1,
    "kp_projection_type": -1,
    "t": "CR",
    "tp": 1,
    "cp": 1,
    "tp_sp": False,
    "kp1": 1,
    "kp2": 1,
}


@dataclass
class MemoryConfig:
    type: str
    scope: str

    @classmethod
    def from_dict(cls, d):
        return cls(
            type=d["type"],
            scope=d["scope"],
        )


@dataclass
class MemoryHierarchyConfig:
    num_levels: int
    mem_hr: list

    @classmethod
    def from_dict(cls, d):
        num_levels = len(d)
        mem_hr = [None] * num_levels
        for level in range(num_levels):
            m = MemoryConfig.from_dict(d["l" + str(level)])
            mem_hr[level] = m
        return cls(
            num_levels=num_levels,
            mem_hr=mem_hr,
        )


ModelLSTMConfig = _namedtuple(
    "model_param",
    [
        "mode",
        "batch_size",
        "vocab_size",
        "num_layers",
        "layer_size",
        "seq_len",
        "projection",
        "num_gates",
        "num_non_linear",
        "num_add",
        "data_scale",
    
    ],
)
GEMMConfig = _namedtuple(
    "model_param",
    [
        "mode",
        "M",
        "K",
        "N",
        "backward",
    ],
)
@dataclass
class LLMAttentionConfig:
    attention_type: str
    num_heads: int
    kv_heads: Optional[int] = None
    use_flashattention: bool = False
    attention_tile_size: Optional[int] = None


@dataclass
class LLMConfig:
    mode: str
    run_type: str
    model_type: str
    tied_embeddings: bool
    num_layers: int
    hidden_dim: int
    global_batch_size: int
    gradient_accumulation_steps: int
    seq_len: int
    decode_len: Optional[int]
    intermediate_size: Optional[int]
    vocab_size: int
    n_tokens: int
    all_reduce: str
    attention: LLMAttentionConfig
    num_experts: int
    top_k: int

    @property
    def num_heads(self) -> int:
        return self.attention.num_heads

    @property
    def use_flashattention(self) -> bool:
        return bool(getattr(self.attention, "use_flashattention", False))

    @property
    def use_moe(self) -> bool:
        return self.num_experts > 1


    @property
    def grad_accumulation_steps(self) -> int:
        """Backward-compatible alias for gradient accumulation steps."""
        return self.gradient_accumulation_steps
LLMInferenceConfig = _namedtuple(
    "inference_param",
    [
        "sample_every",
    ],
)
SWConfig = _namedtuple(
    "sw_param",
    [
        "kernel_launch_overhead",
        "precision",
        "h2d_bandwidth",
        "dp_zero_stage",
        "full_recomputation",
        "dp_microbatch",
    ],
)

SchedulingConfig = _namedtuple(
    "parallelism",
    [
        "auto",
        "dp",
        "lp",
        "mb",
        "kp_hidden_dim1",
        "kp_softmax_dim1",
        "kp_embedding_dim1",
        "kp_projection_dim1",
        "kp_hidden_dim2",
        "kp_softmax_dim2",
        "kp_embedding_dim2",
        "kp_projection_dim2",
        "kp_hidden_type",
        "kp_softmax_type",
        "kp_embedding_type",
        "kp_projection_type",
        "t",
        "tp",
        "cp",
        "tp_sp",
        "kp1",
        "kp2",
    ],
)

FullConfig = _namedtuple(
    "FullConfig",
    [
        "model_config",
        "sw_config",
        "tech_config",
        "power_breakdown",
        "sch_config",
        "area_breakdown",
        "perimeter_breakdown",
        "memory_hierarchy",
        "network_layout",
    ],
)

ExecutionBackendAstraCollectives = _namedtuple(
    "ExecutionBackendAstraCollectives",
    [
        "all_gather",
        "all_reduce",
        "reduce_scatter",
        "all_to_all",
    ],
)

ExecutionBackendAstra = _namedtuple(
    "ExecutionBackendAstra",
    [
        "backend",   # analytical | ns3 | garnet
        "mode",      # hybrid | full_astrasim_hierarchical | full_astrasim_flattened
        "collectives",
        "sys_options",
    ],
)

ExecutionBackendAstraSysOptions = _namedtuple(
    "ExecutionBackendAstraSysOptions",
    [
        "endpoint_delay",
        "active_chunks_per_dimension",
        "preferred_dataset_splits",
    ],
)

ExecutionBackend = _namedtuple(
    "ExecutionBackend",
    [
        "model",   # analytical | astra
        "astra",   # ExecutionBackendAstra or None
    ],
)

InferenceHWConfig = _namedtuple(
    "InferenceHWConfig",
    [
        "kvcache_type",
    ],
)

HWConfig = _namedtuple(
    "HWConfig",
    [
        "sw_config",
        "tech_config",
        "power_breakdown",
        "sch_config",
        "area_breakdown",
        "perimeter_breakdown",
        "memory_hierarchy",
        "network_layout",
        "execution_backend",
        "inference_config",
    ],
)

ModelConfig = _namedtuple(
    "ModelConfig",
    [
        "model_config",
        "inference_config",
    ],
)


def _convert_scalar_string(value: str):
    try:
        return float(value)
    except ValueError:
        pass

    digit = [int(s) for s in value.split() if s.isdigit()]
    order = [str(s) for s in value.split() if not s.isdigit()]
    if not (order and digit):
        return value

    prefix = order[0][0]
    bit = order[0][1] if len(order[0]) > 1 else "B"
    mult = 1

    if prefix == "K":
        mult = 1024
    elif prefix == "M":
        mult = 1024 * 1024
    elif prefix == "G":
        mult = 1024 * 1024 * 1024
    elif prefix == "T":
        mult = 1024 * 1024 * 1024 * 1024
    else:
        raise ValueError(f"Unknown prefix '{prefix}' while parsing value '{value}'")

    if bit == "b":
        mult = mult / 8  # Capacity is expected in Bytes
    elif bit != "B":
        raise ValueError(f"Unknown type '{bit}' while parsing value '{value}'")

    return digit[0] * mult


def _convert_value(value):
    if isinstance(value, dict):
        convert(value)
        return value
    if isinstance(value, list):
        return [_convert_value(item) for item in value]
    if isinstance(value, str):
        try:
            return _convert_scalar_string(value)
        except ValueError:
            return value
    return value


def convert(d):
    if not isinstance(d, dict):
        return d
    for key, val in list(d.items()):
        d[key] = _convert_value(val)
    return d


def parse_bandwidth_string(value):
    """Parse bandwidth/size string (e.g., '300 GB', '1986 GB') to bytes.

    This function uses the same logic as convert() to parse bandwidth strings.
    Returns the numeric value if already a number, or None if value is None.
    """
    if value is None:
        return None

    if not isinstance(value, str):
        return float(value)

    digit = [int(s) for s in value.split() if s.isdigit()]
    order = [str(s) for s in value.split() if not s.isdigit()]

    if not order or not digit:
        # If no units found, try to parse as float
        try:
            return float(value)
        except ValueError:
            raise ValueError(f"Cannot parse bandwidth value: {value}")

    assert len(order) >= 1
    assert len(digit) >= 1

    prefix = order[0][0]
    bit = order[0][1] if len(order[0]) > 1 else 'B'  # Default to Bytes
    mult = 1

    if prefix == "K":
        mult = 1024
    elif prefix == "M":
        mult = 1024 * 1024
    elif prefix == "G":
        mult = 1024 * 1024 * 1024
    elif prefix == "T":
        mult = 1024 * 1024 * 1024 * 1024
    else:
        raise ValueError(f"Unknown prefix: {prefix} in bandwidth value: {value}")

    if bit == "b":
        mult = mult / 8  # Convert bits to Bytes
    elif bit == "B":
        mult = mult
    else:
        raise ValueError(f"Unknown type: {bit} in bandwidth value: {value}")

    return digit[0] * mult


def parse_config(filename, config_type):
    """Parse a yaml configuration file for this experiment.
    Args:
            filename (str): Path to the configuration file
    Returns:
            FullConfig: Contains dataset, model, optimization, training and
            scheduling configurations
    """
    with open(filename, "r") as f:
        try:
            config_dict = _yaml.safe_load(f)
        except _YAMLError as exc:
            hint = (
                f"Failed to parse YAML config '{filename}'. "
                "Please check indentation and required sections like 'attention' and parameters such as 'num_experts'."
            )
            raise ValueError(hint) from exc
        # print(config_dict)
        convert(config_dict)
    if config_type == "hardware":
        sw_block = dict(config_dict["sw_param"])
        precision_spec = sw_block["precision"]
        precision_config = _parse_precision_block(precision_spec)
        kernel_launch_overhead = sw_block["kernel_launch_overhead"]
        h2d_bandwidth = sw_block["h2d_bandwidth"]
        dp_zero_stage = sw_block["dp_zero_stage"]
        full_recomp_raw = sw_block.get("full_recomputation", False)
        if isinstance(full_recomp_raw, str):
            full_recomputation = full_recomp_raw.strip().lower() in {"1", "true", "yes", "y", "on", "full"}
        else:
            full_recomputation = bool(full_recomp_raw)
        dp_microbatch_raw = sw_block.get("dp_microbatch", "every_mb")
        if isinstance(dp_microbatch_raw, str):
            dp_microbatch = dp_microbatch_raw.strip().lower()
        else:
            dp_microbatch = str(dp_microbatch_raw).strip().lower()
        if dp_microbatch not in {"every_mb", "last_mb"}:
            raise ValueError("sw_param.dp_microbatch must be 'every_mb' or 'last_mb'")
        sw_config = SWConfig(
            kernel_launch_overhead=kernel_launch_overhead,
            precision=precision_config,
            h2d_bandwidth=h2d_bandwidth,
            dp_zero_stage=dp_zero_stage,
            full_recomputation=full_recomputation,
            dp_microbatch=dp_microbatch,
        )
        raw_parallelism_block = config_dict.get("parallelism", {})
        if not isinstance(raw_parallelism_block, dict):
            raise ValueError("parallelism must be a mapping")
        parallelism_params = dict(_PARALLELISM_DEFAULTS)
        parallelism_params.update(raw_parallelism_block)
        parallelism_params["auto"] = bool(parallelism_params["auto"])
        parallelism_params["tp_sp"] = bool(parallelism_params["tp_sp"])
        sch_config = SchedulingConfig(**parallelism_params)
        scheduling_for_network = {str(k).lower(): v for k, v in parallelism_params.items()}

        network_spec = config_dict.get("network")
        network_dimensions, network_faults, network_overlap = _parse_network_layout(network_spec, scheduling_for_network)
        if not network_dimensions:
            raise ValueError("network section must define at least one dimension")
        network_layout_config = _build_network_layout_config(network_dimensions, network_faults, network_overlap)

        tech_config = TechConfig.from_dict(config_dict["tech_param"])

        # Optional breakdown configs (not needed for simplified configs)
        if "power_breakdown" in config_dict:
            power_config = PowerBreakdownConfig.from_dict(config_dict["power_breakdown"])
        else:
            # Create dummy power config with zeros
            power_config = PowerBreakdownConfig(
                TDP=1.0, core=1.0, DRAM=1.0, L2=1.0, L1=1.0, reg_mem=1.0,
                network=NetworkPowerConfig(inter_node=1.0, intra_node=1.0)
            )

        if "area_breakdown" in config_dict:
            area_config = AreaBreakdownConfig.from_dict(config_dict["area_breakdown"])
        else:
            # Create dummy area config with zeros
            area_config = AreaBreakdownConfig(
                proc_chip_area_budget=1.0, core=1.0, DRAM=1.0, L2=1.0, L1=1.0, reg_mem=1.0,
                node_area_budget=1.0, network=NetworkAreaConfig(inter_node=1.0, intra_node=1.0)
            )

        if "perimeter_breakdown" in config_dict:
            perimeter_config = PerimeterBreakdownConfig.from_dict(config_dict["perimeter_breakdown"])
        else:
            # Create dummy perimeter config with zeros
            perimeter_config = PerimeterBreakdownConfig(DRAM=0.1, inter_node=0.1, intra_node=0.1)

        memory_hierarchy_config = MemoryHierarchyConfig.from_dict(
            config_dict["memory_hierarchy"]
        )
        # execution backend (optional)
        eb_dict = config_dict.get("execution_backend", {})
        eb_model = eb_dict.get("model", "analytical")
        astra_cfg = eb_dict.get("astra", {}) if eb_model == "astra" else None
        if astra_cfg is not None:
            coll = astra_cfg.get("collectives", {})
            coll_cfg = ExecutionBackendAstraCollectives(
                all_gather=coll.get("all_gather", "auto"),
                all_reduce=coll.get("all_reduce", "auto"),
                reduce_scatter=coll.get("reduce_scatter", "auto"),
                all_to_all=coll.get("all_to_all", "auto"),
            )
            sys_cfg_dict = astra_cfg.get("sys_options")
            if sys_cfg_dict is not None:
                sys_cfg = ExecutionBackendAstraSysOptions(
                    endpoint_delay=sys_cfg_dict.get("endpoint_delay"),
                    active_chunks_per_dimension=sys_cfg_dict.get(
                        "active_chunks_per_dimension"
                    ),
                    preferred_dataset_splits=sys_cfg_dict.get(
                        "preferred_dataset_splits"
                    ),
                )
            else:
                sys_cfg = None
            eb_astra = ExecutionBackendAstra(
                backend=astra_cfg.get("backend", "analytical"),
                mode=astra_cfg.get("mode", "hybrid"),
                collectives=coll_cfg,
                sys_options=sys_cfg,
            )
        else:
            eb_astra = None
        exec_backend = ExecutionBackend(model=eb_model, astra=eb_astra)

        inference_dict = config_dict.get("inference", {}) or {}
        inference_cfg = InferenceHWConfig(
            kvcache_type=inference_dict.get("kvcache_type", "hbm_only"),
        )

        config = HWConfig(
            sw_config=sw_config,
            tech_config=tech_config,
            power_breakdown=power_config,
            sch_config=sch_config,
            area_breakdown=area_config,
            perimeter_breakdown=perimeter_config,
            memory_hierarchy=memory_hierarchy_config,
            network_layout=network_layout_config,
            execution_backend=exec_backend,
            inference_config=inference_cfg,
        )
    elif config_type == "LSTM":
        model_config = ModelLSTMConfig(**config_dict["model_param"])
        config = ModelConfig(model_config=model_config, inference_config=None)
    elif config_type == "GEMM":
        mp = dict(config_dict["model_param"])  # copy
        if "backward" not in mp:
            mp["backward"] = False
        model_config = GEMMConfig(**mp)
        config = ModelConfig(model_config=model_config, inference_config=None)
    elif config_type == "LLM":
        mp = dict(config_dict["model_param"])
        if "attention" not in mp:
            raise ValueError("model_param.attention section must be specified for LLM configs")
        attention_dict = mp.pop("attention")
        if not isinstance(attention_dict, dict):
            raise ValueError("model_param.attention must be a mapping of attention parameters")

        if "attention_type" not in attention_dict:
            raise ValueError("model_param.attention.attention_type must be specified")
        attn_type_raw = attention_dict["attention_type"]
        attn_type = str(attn_type_raw).strip().lower()
        if attn_type == "mla":
            raise NotImplementedError("attention_type='mla' is not yet supported. Please use 'mha' or 'gqa'.")
        if attn_type not in {"mha", "gqa"}:
            raise ValueError(
                f"model_param.attention.attention_type must be either 'mha' or 'gqa' (got {attn_type_raw!r})"
            )

        if "num_heads" not in attention_dict:
            raise ValueError("model_param.attention.num_heads must be specified")
        try:
            num_heads = int(attention_dict["num_heads"])
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"model_param.attention.num_heads must be an integer (got {attention_dict['num_heads']!r})"
            ) from exc
        if num_heads <= 0:
            raise ValueError("model_param.attention.num_heads must be a positive integer")

        kv_heads_field = attention_dict.get("kv_heads")

        if attn_type == "gqa":
            if kv_heads_field is None:
                raise ValueError(
                    "model_param.attention.kv_heads must be specified when attention_type='gqa'"
                )
            try:
                kv_heads = int(kv_heads_field)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"model_param.attention.kv_heads must be an integer when attention_type='gqa' (got {kv_heads_field!r})"
                ) from exc
        else:
            # For MHA/MLA, ignore any provided kv_heads override and default to num_heads.
            kv_heads = num_heads

        if kv_heads <= 0:
            raise ValueError("model_param.attention.kv_heads must be a positive integer")
        if num_heads % kv_heads != 0:
            raise ValueError(
                f"model_param.attention.kv_heads={kv_heads} must divide num_heads={num_heads}"
            )

        raw_flash_attention = attention_dict.get("use_flashattention", attention_dict.get("used_flash_attention", False))
        if isinstance(raw_flash_attention, str):
            flash_attention = raw_flash_attention.strip().lower() in {"1", "true", "yes", "y"}
        else:
            flash_attention = bool(raw_flash_attention)

        tile_size_field = attention_dict.get("attention_tile_size")
        if flash_attention:
            if tile_size_field is None:
                raise ValueError(
                    "model_param.attention.attention_tile_size must be specified when flash attention is enabled"
                )
            try:
                attention_tile_size = int(tile_size_field)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"model_param.attention.attention_tile_size must be an integer when flash attention is enabled (got {tile_size_field!r})"
                ) from exc
            if attention_tile_size <= 0:
                raise ValueError("model_param.attention.attention_tile_size must be a positive integer when flash attention is enabled")
        else:
            # Flash attention disabled; tile size, if provided, is ignored.
            attention_tile_size = None

        attention_cfg = LLMAttentionConfig(
            attention_type=attn_type,
            num_heads=num_heads,
            kv_heads=kv_heads,
            use_flashattention=flash_attention,
            attention_tile_size=attention_tile_size,
        )

        num_experts_field = mp.pop("num_experts", None)
        if num_experts_field is None:
            raise ValueError("model_param.num_experts must be specified for LLM configs")
        try:
            num_experts = int(num_experts_field)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"model_param.num_experts must be an integer (got {num_experts_field!r})"
            ) from exc
        if num_experts <= 0:
            raise ValueError("model_param.num_experts must be a positive integer")

        top_k_field = mp.pop("top_k", None)
        if top_k_field is None:
            top_k = 1
        else:
            try:
                top_k = int(top_k_field)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"model_param.top_k must be an integer when provided (got {top_k_field!r})"
                ) from exc
            if top_k <= 0:
                raise ValueError("model_param.top_k must be a positive integer when provided")


        if "run_type" not in mp:
            raise ValueError("model_param.run_type must be specified")
        run_type_raw = mp.pop("run_type")
        run_type = str(run_type_raw).strip().lower()
        if run_type not in {"training", "inference"}:
            raise ValueError(f"model_param.run_type must be either 'training' or 'inference' (got {run_type_raw!r})")
        if "tied_embeddings" not in mp:
            raise ValueError("model_param.tied_embeddings must be specified")
        tied_embeddings_raw = mp.pop("tied_embeddings")
        if isinstance(tied_embeddings_raw, str):
            tied_embeddings = tied_embeddings_raw.strip().lower() in {"1", "true", "yes", "y"}
        else:
            tied_embeddings = bool(tied_embeddings_raw)
        if "model_type" not in mp:
            raise ValueError("model_param.model_type must be specified")
        model_type_raw = mp.pop("model_type")
        model_type = str(model_type_raw).strip().lower()
        if model_type not in {"gpt", "llama"}:
            raise ValueError(
                f"model_param.model_type must be either 'gpt' or 'llama' (got {model_type_raw!r})"
            )
        decode_len = mp.pop("decode_len", None)
        if decode_len is not None:
            try:
                decode_len = int(decode_len)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"model_param.decode_len must be an integer when provided (got {decode_len!r})") from exc

        inference_config = None
        if run_type == "inference":
            if decode_len is None:
                raise ValueError("model_param.decode_len must be specified when run_type is 'inference'")
            inference_dict = dict(config_dict.get("inference_param", {}) or {})
            sample_every_raw = inference_dict.get("sample_every", -1)
            try:
                sample_every = int(sample_every_raw)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"inference_param.sample_every must be an integer (got {sample_every_raw!r})"
                ) from exc
            inference_config = LLMInferenceConfig(sample_every=sample_every)

        if "mode" not in mp:
            raise ValueError("model_param.mode must be specified")
        mode_raw = mp.pop("mode")
        mode = str(mode_raw).strip().upper()
        if mode != "LLM":
            raise ValueError(f"model_param.mode must be 'LLM' for LLM configs (got {mode_raw!r})")

        def _pop_required_int(field: str) -> int:
            if field not in mp:
                raise ValueError(f"model_param.{field} must be specified")
            raw_value = mp.pop(field)
            try:
                value = int(raw_value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"model_param.{field} must be an integer (got {raw_value!r})") from exc
            if value <= 0:
                raise ValueError(f"model_param.{field} must be a positive integer")
            return value

        num_layers = _pop_required_int("num_layers")
        hidden_dim = _pop_required_int("hidden_dim")

        global_batch_size = _pop_required_int("global_batch_size")


        grad_accum_field = mp.pop("gradient_accumulation_steps", None)
        if grad_accum_field is None:
            grad_accum_field = mp.pop("gradient_accumulation_step", None)
        if grad_accum_field is None:
            gradient_accumulation_steps = 1
        else:
            try:
                gradient_accumulation_steps = int(grad_accum_field)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"model_param.gradient_accumulation_steps must be an integer (got {grad_accum_field!r})"
                ) from exc
            if gradient_accumulation_steps <= 0:
                raise ValueError("model_param.gradient_accumulation_steps must be a positive integer")

        seq_len = _pop_required_int("seq_len")
        vocab_size = _pop_required_int("vocab_size")

        intermediate_size = mp.pop("intermediate_size", None)
        if intermediate_size is None:
            raise ValueError("model_param.intermediate_size must be specified for LLM configs")
        try:
            intermediate_size = int(intermediate_size)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"model_param.intermediate_size must be an integer (got {intermediate_size!r})") from exc
        if intermediate_size <= 0:
            raise ValueError("model_param.intermediate_size must be a positive integer")

        model_config = LLMConfig(
            mode=mode,
            run_type=run_type,
            model_type=model_type,
            tied_embeddings=tied_embeddings,
            num_layers=num_layers,
            hidden_dim=hidden_dim,
            global_batch_size=global_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            seq_len=seq_len,
            decode_len=decode_len,
            intermediate_size=intermediate_size,
            vocab_size=vocab_size,
            n_tokens=0, # not used for now.
            all_reduce="every layer", # hard set for now.
            attention=attention_cfg,
            num_experts=num_experts,
            top_k=top_k,
        )
        config = ModelConfig(model_config=model_config, inference_config=inference_config)
    else:
        raise ValueError("Invalid config type: {}".format(config_type))
    
    # model_config = ModelConfig(**config_dict["model_param"])
    # sw_config = SWConfig(**config_dict["sw_param"])
    # sch_config = SchedulingConfig(**config_dict["parallelism"])
    # tech_config = TechConfig.from_dict(config_dict["tech_param"])
    # power_config = PowerBreakdownConfig.from_dict(config_dict["power_breakdown"])
    # area_config = AreaBreakdownConfig.from_dict(config_dict["area_breakdown"])
    # perimeter_config = PerimeterBreakdownConfig.from_dict(
    #     config_dict["perimeter_breakdown"]
    # )
    # system_config = SystemHierarchyConfig.from_dict(config_dict["system_hierarchy"])
    # memory_hierarchy_config = MemoryHierarchyConfig.from_dict(
    #     config_dict["memory_hierarchy"]
    # )
    # network_topology_config = NetworkTopologyConfig.from_dict(
    #     config_dict["network_topology"]
    # )

    return config
