"""Parquetファイルの読み書きを担当するクラス。"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq


class ParquetStore:
    """Parquetファイルの読み書きのみを担当。データの意味は知らない。"""

    def __init__(self, data_dir: str = "data") -> None:
        self.data_dir = Path(data_dir)

    def read(self, category: str, name: str, filters: list[tuple] | None = None) -> pd.DataFrame:
        """Parquetを読み取る。

        Args:
            category: カテゴリ (例: "raw", "odds")
            name: テーブル名 (例: "races")
            filters: pyarrow述語プッシュダウン用フィルタ
                     [(column, op, value), ...] 例: [("race_date", ">=", dt)]
        """
        path = self.data_dir / category / name
        if path.is_dir():
            dataset = ds.dataset(str(path), format="parquet", partitioning="hive")
            if filters:
                mask = None
                for col, op, val in filters:
                    if op == ">=":
                        cond = ds.field(col) >= val
                    elif op == "<=":
                        cond = ds.field(col) <= val
                    elif op == "==":
                        cond = ds.field(col) == val
                    else:
                        raise ValueError(f"Unsupported filter operator: {op}")
                    mask = cond if mask is None else mask & cond
                table = dataset.to_table(filter=mask)
            else:
                table = dataset.to_table()
            return table.to_pandas()
        return pd.read_parquet(path.with_suffix(".parquet"), filters=filters)

    def write(
        self,
        category: str,
        name: str,
        df: pd.DataFrame,
        partition_cols: list[str] | None = None,
    ) -> None:
        """DataFrameをParquetに書き込む。

        partition_cols未指定時は単一ファイル（アトミック書き込み）。
        指定時はパーティション書き込み。
        """
        path = self.data_dir / category / name
        path.parent.mkdir(parents=True, exist_ok=True)

        if partition_cols:
            table = pa.Table.from_pandas(df)
            pq.write_to_dataset(table, root_path=str(path), partition_cols=partition_cols)
        else:
            file_path = path.with_suffix(".parquet")
            tmp = file_path.with_suffix(".parquet.tmp")
            df.to_parquet(tmp, index=False)
            tmp.replace(file_path)

    def exists(self, category: str, name: str) -> bool:
        """ファイル or パーティションディレクトリが存在するか。"""
        path = self.data_dir / category / name
        return path.with_suffix(".parquet").exists() or path.is_dir()
