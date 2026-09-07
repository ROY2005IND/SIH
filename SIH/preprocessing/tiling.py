"""
Spatial Tiling Engine for Gigapixel Lunar Rasters.
Provides memory-aware chunking with boundary margins to avoid match dropout near seams.
"""
from dataclasses import dataclass
from typing import Iterator
import numpy as np

@dataclass
class TileMetadata:
    """Metadata describing a spatial image tile and its global coordinate offset."""
    tile_id: int
    row: int
    col: int
    x_min: int
    y_min: int
    x_max: int
    y_max: int
    global_x_offset: int
    global_y_offset: int
    tile_image: np.ndarray

class RasterTiler:
    """Tiling manager dividing large rasters into overlapping chunks."""

    def __init__(
        self,
        image: np.ndarray,
        tile_size: int = 1024,
        overlap_px: int = 128,
    ):
        self.image = image
        self.h, self.w = image.shape[:2]
        self.tile_size = tile_size
        self.overlap = overlap_px

    def generate_tiles(self) -> Iterator[TileMetadata]:
        """Yield tiles sequentially with boundary overlap to ensure continuous feature coverage."""
        stride = max(self.tile_size - self.overlap, 64)
        tile_id = 0

        for r_idx, y in enumerate(range(0, self.h, stride)):
            y_end = min(y + self.tile_size, self.h)
            for c_idx, x in enumerate(range(0, self.w, stride)):
                x_end = min(x + self.tile_size, self.w)
                tile_arr = self.image[y:y_end, x:x_end]

                yield TileMetadata(
                    tile_id=tile_id,
                    row=r_idx,
                    col=c_idx,
                    x_min=x,
                    y_min=y,
                    x_max=x_end,
                    y_max=y_end,
                    global_x_offset=x,
                    global_y_offset=y,
                    tile_image=tile_arr,
                )
                tile_id += 1

    @staticmethod
    def map_tile_points_to_global(
        points_local: np.ndarray,
        tile: TileMetadata,
    ) -> np.ndarray:
        """Offset local tile coordinate points back to global image coordinates."""
        if len(points_local) == 0:
            return np.empty((0, 2), dtype=np.float32)
        offset = np.array([tile.global_x_offset, tile.global_y_offset], dtype=np.float32)
        return points_local + offset
