"""
清除 parquet 文件中的 progress_predicted 相关列
"""
import argparse
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import re


def clean_parquet_files(source_path: Path, column_pattern: str = None, dry_run: bool = False):
    """
    清除 parquet 文件中匹配的列
    
    Args:
        source_path: 数据集路径
        column_pattern: 要删除的列名模式（正则表达式），默认匹配所有 progress_predicted 开头的列
        dry_run: 如果为 True，只显示会删除的列，不实际删除
    """
    data_parquet = source_path / "data"
    
    if not data_parquet.exists():
        print(f"错误: 数据目录不存在: {data_parquet}")
        return
    
    # 默认匹配所有 progress_predicted 开头的列
    if column_pattern is None:
        column_pattern = r"^progress_predicted.*"
    
    pattern = re.compile(column_pattern)
    
    # 获取所有 parquet 文件
    parquet_files = list(data_parquet.glob("**/*.parquet"))
    print(f"找到 {len(parquet_files)} 个 parquet 文件")
    
    if len(parquet_files) == 0:
        print("没有找到 parquet 文件")
        return
    
    # 先检查第一个文件，看看有哪些匹配的列
    sample_df = pd.read_parquet(parquet_files[0])
    matching_cols = [col for col in sample_df.columns if pattern.match(col)]
    
    if not matching_cols:
        print(f"没有找到匹配模式 '{column_pattern}' 的列")
        print(f"现有列: {list(sample_df.columns)}")
        return
    
    print(f"将删除以下列: {matching_cols}")
    
    if dry_run:
        print("\n[DRY RUN] 不会实际删除，只是预览")
        return
    
    # 遍历所有文件并删除匹配的列
    cleaned_count = 0
    for parquet_path in tqdm(parquet_files, desc="清理中"):
        df = pd.read_parquet(parquet_path)
        cols_to_drop = [col for col in df.columns if pattern.match(col)]
        
        if cols_to_drop:
            df = df.drop(columns=cols_to_drop)
            df.to_parquet(parquet_path, index=False)
            cleaned_count += 1
    
    print(f"\n清理完成，共处理 {cleaned_count} 个文件")


def main():
    parser = argparse.ArgumentParser(description="清除 parquet 文件中的 progress_predicted 相关列")
    parser.add_argument("--source_path", type=str, required=True, help="数据集路径")
    parser.add_argument("--column_pattern", type=str, default=None, 
                        help="要删除的列名模式（正则表达式），默认匹配所有 progress_predicted 开头的列")
    parser.add_argument("--exact_column", type=str, default=None,
                        help="要删除的精确列名（优先于 column_pattern）")
    parser.add_argument("--dry_run", action="store_true", help="只预览不实际删除")
    
    args = parser.parse_args()
    
    # 如果指定了精确列名，转换为正则表达式
    if args.exact_column:
        args.column_pattern = f"^{re.escape(args.exact_column)}$"
    
    clean_parquet_files(
        source_path=Path(args.source_path),
        column_pattern=args.column_pattern,
        dry_run=args.dry_run
    )


if __name__ == "__main__":
    main()
