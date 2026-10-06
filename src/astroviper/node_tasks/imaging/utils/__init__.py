"""Node-task imaging utilities (experimental I/O variants)."""

from astroviper.node_tasks.imaging.utils.skunk_works import (
    compute_shard_task_priorities,
    load_processing_set_skunk_works,
    precreate_sharded_files,
    read_array_region,
    record_shard_ost_map,
    write_result_chunk_to_disk_sharded_skunk_works,
    write_result_chunk_to_disk_using_zarr_skunk_works,
)
from astroviper.node_tasks.imaging.utils.skunk_works_fits import (
    create_empty_fits_images,
    write_result_chunk_to_fits_skunk_works,
)
from astroviper.node_tasks.imaging.utils.skunk_works_msv2 import (
    add_lazy_input_data,
    check_data_group_skunk_works_msv2,
    is_fatal_load_error,
    load_processing_set_skunk_works_msv2,
    msv2_engine_available,
    open_processing_set_skunk_works_msv2,
    require_msv2_engine,
)

__all__ = [
    "add_lazy_input_data",
    "check_data_group_skunk_works_msv2",
    "compute_shard_task_priorities",
    "create_empty_fits_images",
    "is_fatal_load_error",
    "load_processing_set_skunk_works",
    "load_processing_set_skunk_works_msv2",
    "msv2_engine_available",
    "open_processing_set_skunk_works_msv2",
    "read_array_region",
    "record_shard_ost_map",
    "require_msv2_engine",
    "write_result_chunk_to_disk_using_zarr_skunk_works",
    "write_result_chunk_to_disk_sharded_skunk_works",
    "write_result_chunk_to_fits_skunk_works",
    "precreate_sharded_files",
]
