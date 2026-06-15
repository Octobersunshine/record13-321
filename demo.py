import pandas as pd
from group_ranking import GroupRankingService


def create_sample_data():
    data = {
        '员工': ['张三', '李四', '王五', '赵六', '钱七', '孙八', '周九', '吴十',
                '郑十一', '王十二', '冯十三', '陈十四'],
        '部门': ['销售部', '销售部', '销售部', '销售部', '销售部',
                '技术部', '技术部', '技术部', '技术部',
                '市场部', '市场部', '市场部'],
        '销售额': [120, 150, 150, 100, 80,
                   200, 180, 180, 180,
                   90, 110, 110],
        '完成率': [0.85, 0.92, 0.92, 0.78, 0.65,
                   0.95, 0.88, 0.91, 0.88,
                   0.72, 0.83, 0.85],
    }
    return pd.DataFrame(data)


def demo_basic_ranking():
    print("=" * 70)
    print("【示例1】基础分组排名 - 各部门内按销售额排名（min方式）")
    print("=" * 70)
    service = GroupRankingService()
    df = create_sample_data()
    result = service.rank(
        df=df,
        group_cols='部门',
        value_col='销售额',
        method='min',
        ascending=False,
    )
    print(result.sort_values(['部门', 'rank']).to_string(index=False))
    print()


def demo_compare_methods():
    print("=" * 70)
    print("【示例2】对比不同并列处理方式（以销售部为例）")
    print("=" * 70)
    service = GroupRankingService()
    df = create_sample_data()
    sales_df = df[df['部门'] == '销售部'].copy()

    print("销售部原始数据：")
    print(sales_df[['员工', '销售额']].to_string(index=False))
    print()

    methods = ['min', 'max', 'dense', 'average', 'first']
    for method in methods:
        result = service.rank(
            df=sales_df,
            group_cols='部门',
            value_col='销售额',
            rank_col_name=f'排名({method})',
            method=method,
            ascending=False,
        )
        print(f"--- {method} 方式 ---")
        print(result[['员工', '销售额', f'排名({method})']].to_string(index=False))
        print()


def demo_multi_columns():
    print("=" * 70)
    print("【示例3】多列同时排名 - 按销售额降序、完成率降序")
    print("=" * 70)
    service = GroupRankingService()
    df = create_sample_data()
    result = service.rank_multi_columns(
        df=df,
        group_cols='部门',
        value_cols=['销售额', '完成率'],
        method='dense',
        ascending=[False, False],
    )
    print(result.sort_values(['部门', '销售额_rank']).to_string(index=False))
    print()


def demo_top_n():
    print("=" * 70)
    print("【示例4】各部门前2名（Top N 筛选）")
    print("=" * 70)
    service = GroupRankingService()
    df = create_sample_data()
    result = service.get_rank_summary(
        df=df,
        group_cols='部门',
        value_col='销售额',
        method='min',
        ascending=False,
        top_n=2,
    )
    print(result.to_string(index=False))
    print()


def demo_ascending():
    print("=" * 70)
    print("【示例5】升序排名 - 各部门内完成率从低到高（数值小的排前面）")
    print("=" * 70)
    service = GroupRankingService()
    df = create_sample_data()
    result = service.rank(
        df=df,
        group_cols='部门',
        value_col='完成率',
        rank_col_name='倒数排名',
        method='min',
        ascending=True,
    )
    print(result.sort_values(['部门', '倒数排名']).to_string(index=False))
    print()


def demo_pct_rank():
    print("=" * 70)
    print("【示例6】百分比排名（pct=True）- 各部门内销售额百分位")
    print("=" * 70)
    service = GroupRankingService()
    df = create_sample_data()
    result = service.rank(
        df=df,
        group_cols='部门',
        value_col='销售额',
        rank_col_name='销售额百分位',
        method='average',
        ascending=False,
        pct=True,
    )
    result['销售额百分位'] = result['销售额百分位'].apply(lambda x: f"{x:.1%}")
    print(result.sort_values(['部门', '销售额百分位']).to_string(index=False))
    print()


def create_sample_data_with_nulls():
    data = {
        '员工': ['张三', '李四', '王五', '赵六', '钱七', '孙八', '周九', '吴十',
                '郑十一', '王十二', '冯十三', '陈十四'],
        '部门': ['销售部', '销售部', '销售部', '销售部', '销售部',
                '技术部', '技术部', '技术部', '技术部',
                '市场部', '市场部', '市场部'],
        '销售额': [120, None, 150, None, 80,
                   200, None, 180, 180,
                   90, 110, None],
        '完成率': [0.85, None, 0.92, 0.78, 0.65,
                   0.95, 0.88, None, 0.88,
                   0.72, None, 0.85],
    }
    return pd.DataFrame(data)


def demo_null_ranking():
    print("=" * 70)
    print("【示例7】含空值排名 - 空值统一排最后（na_option='bottom'）")
    print("=" * 70)
    service = GroupRankingService()
    df = create_sample_data_with_nulls()
    print("含空值的原始数据：")
    print(df.to_string(index=False))
    print()

    for method in ['min', 'max', 'dense', 'first']:
        result = service.rank(
            df=df,
            group_cols='部门',
            value_col='销售额',
            rank_col_name=f'排名({method})',
            method=method,
            ascending=False,
        )
        print(f"--- {method} 方式 ---")
        print(result[['员工', '部门', '销售额', f'排名({method})']].to_string(index=False))
        print()

    print("--- average 方式 ---")
    result = service.rank(
        df=df,
        group_cols='部门',
        value_col='销售额',
        rank_col_name='排名(average)',
        method='average',
        ascending=False,
    )
    print(result[['员工', '部门', '销售额', '排名(average)']].to_string(index=False))
    print()


def demo_null_multi_columns():
    print("=" * 70)
    print("【示例8】含空值的多列排名 - 空值统一排最后")
    print("=" * 70)
    service = GroupRankingService()
    df = create_sample_data_with_nulls()
    result = service.rank_multi_columns(
        df=df,
        group_cols='部门',
        value_cols=['销售额', '完成率'],
        method='dense',
        ascending=[False, False],
    )
    print(result.sort_values(['部门', '销售额_rank'], na_position='last').to_string(index=False))
    print()


def demo_null_top_n():
    print("=" * 70)
    print("【示例9】含空值的 Top N 筛选 - 空值不在 Top N 内")
    print("=" * 70)
    service = GroupRankingService()
    df = create_sample_data_with_nulls()
    result = service.get_rank_summary(
        df=df,
        group_cols='部门',
        value_col='销售额',
        method='min',
        ascending=False,
        top_n=2,
    )
    print(result.to_string(index=False))
    print()


def demo_percentile_basic():
    print("=" * 70)
    print("【示例10】分组百分位 - 各部门内销售额百分位（值越高百分位越高）")
    print("=" * 70)
    service = GroupRankingService()
    df = create_sample_data()
    result = service.percentile(
        df=df,
        group_cols='部门',
        value_col='销售额',
        method='average',
        ascending=True,
        scale=100.0,
    )
    result['销售额百分位'] = result['percentile'].apply(lambda x: f"{x:.1f}" if pd.notna(x) else "NaN")
    display_cols = ['员工', '部门', '销售额', '销售额百分位']
    print(result.sort_values(['部门', '销售额'], ascending=[True, False])[display_cols].to_string(index=False))
    print()


def demo_percentile_compare_methods():
    print("=" * 70)
    print("【示例11】不同并列方式下的百分位对比（以销售部为例）")
    print("=" * 70)
    service = GroupRankingService()
    df = create_sample_data()
    sales_df = df[df['部门'] == '销售部'].copy()

    methods = ['min', 'max', 'average', 'dense', 'first']
    result = sales_df[['员工', '销售额']].copy()

    for method in methods:
        pct_df = service.percentile(
            df=sales_df,
            group_cols='部门',
            value_col='销售额',
            percentile_col_name=f'百分位({method})',
            method=method,
            ascending=True,
            scale=100.0,
        )
        result[f'百分位({method})'] = pct_df[f'百分位({method})'].apply(
            lambda x: f"{x:.1f}" if pd.notna(x) else "NaN"
        )

    print(result.sort_values('销售额', ascending=False).to_string(index=False))
    print()


def demo_percentile_descending():
    print("=" * 70)
    print("【示例12】降序百分位 - 完成率越低百分位越高（倒数百分位）")
    print("=" * 70)
    service = GroupRankingService()
    df = create_sample_data()
    result = service.percentile(
        df=df,
        group_cols='部门',
        value_col='完成率',
        percentile_col_name='完成率_倒数百分位',
        method='average',
        ascending=False,
        scale=100.0,
    )
    result['完成率_倒数百分位_fmt'] = result['完成率_倒数百分位'].apply(
        lambda x: f"{x:.1f}" if pd.notna(x) else "NaN"
    )
    display_cols = ['员工', '部门', '完成率', '完成率_倒数百分位_fmt']
    print(result.sort_values(['部门', '完成率'])[display_cols].to_string(index=False))
    print()


def demo_percentile_multi_columns():
    print("=" * 70)
    print("【示例13】多列百分位同时计算 - 销售额和完成率")
    print("=" * 70)
    service = GroupRankingService()
    df = create_sample_data()
    result = service.percentile_multi_columns(
        df=df,
        group_cols='部门',
        value_cols=['销售额', '完成率'],
        percentile_col_suffix='_百分位',
        method='average',
        ascending=[True, True],
        scale=100.0,
    )
    result['销售额_百分位_fmt'] = result['销售额_百分位'].apply(
        lambda x: f"{x:.1f}" if pd.notna(x) else "NaN"
    )
    result['完成率_百分位_fmt'] = result['完成率_百分位'].apply(
        lambda x: f"{x:.1f}" if pd.notna(x) else "NaN"
    )
    display_cols = ['员工', '部门', '销售额', '销售额_百分位_fmt', '完成率', '完成率_百分位_fmt']
    print(result.sort_values(['部门', '销售额'], ascending=[True, False])[display_cols].to_string(index=False))
    print()


def demo_percentile_with_nulls():
    print("=" * 70)
    print("【示例14】含空值的百分位 - 空值排最后（百分位最低）")
    print("=" * 70)
    service = GroupRankingService()
    df = create_sample_data_with_nulls()
    result = service.percentile(
        df=df,
        group_cols='部门',
        value_col='销售额',
        method='average',
        ascending=True,
        na_option='bottom',
        scale=100.0,
    )
    result['销售额百分位'] = result['percentile'].apply(
        lambda x: f"{x:.1f}" if pd.notna(x) else "NaN"
    )
    display_cols = ['员工', '部门', '销售额', '销售额百分位']
    print(result.sort_values(['部门', '销售额'], ascending=[True, False], na_position='last')[display_cols].to_string(index=False))
    print()


if __name__ == '__main__':
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 200)
    pd.set_option('display.max_colwidth', 30)

    demo_basic_ranking()
    demo_compare_methods()
    demo_multi_columns()
    demo_top_n()
    demo_ascending()
    demo_pct_rank()
    demo_null_ranking()
    demo_null_multi_columns()
    demo_null_top_n()
    demo_percentile_basic()
    demo_percentile_compare_methods()
    demo_percentile_descending()
    demo_percentile_multi_columns()
    demo_percentile_with_nulls()
