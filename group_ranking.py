import pandas as pd
from typing import Optional, Union, List


class GroupRankingService:
    """
    分组排名与百分位服务：按分组（如部门内）对数值列进行排名和百分位计算，
    支持多种并列处理方式。

    支持的并列处理方式 (method 参数)：
        - 'average': 并列名次取平均值（如 1, 2.5, 2.5, 4）
        - 'min': 并列名次取最小值（如 1, 2, 2, 4），对应标准比赛排名
        - 'max': 并列名次取最大值（如 1, 3, 3, 4）
        - 'dense': 密集排名，不跳过名次（如 1, 2, 2, 3）
        - 'first': 按出现顺序分配不同名次（值相同的按原始顺序排名）
    """

    VALID_METHODS = {'average', 'min', 'max', 'dense', 'first'}
    VALID_ASCENDING_OPTIONS = {True, False}
    VALID_NA_OPTIONS = {'keep', 'top', 'bottom'}

    def __init__(self):
        pass

    def rank(
        self,
        df: pd.DataFrame,
        group_cols: Union[str, List[str]],
        value_col: str,
        rank_col_name: str = 'rank',
        method: str = 'min',
        ascending: bool = False,
        na_option: str = 'bottom',
        pct: bool = False,
        dropna: bool = True,
    ) -> pd.DataFrame:
        """
        对 DataFrame 按分组列进行分组排名。

        参数:
            df: 输入的 DataFrame
            group_cols: 分组列名或列名列表（如 '部门' 或 ['部门', '组别']）
            value_col: 要排名的数值列名（如 '销售额'）
            rank_col_name: 输出的排名列名，默认为 'rank'
            method: 并列处理方式，可选 'average' | 'min' | 'max' | 'dense' | 'first'
            ascending: 是否升序排名。False=值越大排名越靠前（适用于销售额、得分等），
                       True=值越小排名越靠前（适用于耗时、成本等）
            na_option: 空值处理方式，默认 'bottom'（空值统一排最后），可选 'keep' | 'top' | 'bottom'
            pct: 是否以百分比形式显示排名
            dropna: 排名计算时是否排除空值

        返回:
            新增了排名列的 DataFrame（副本，不修改原数据）
        """
        self._validate_params(df, group_cols, value_col, method, ascending, na_option)

        result_df = df.copy()

        if isinstance(group_cols, str):
            group_cols = [group_cols]

        ranked_series = result_df.groupby(group_cols, dropna=dropna)[value_col].rank(
            method=method,
            ascending=ascending,
            na_option=na_option,
            pct=pct,
        )
        if method in {'min', 'max', 'dense', 'first'} and not pct:
            ranked_series = ranked_series.astype('Int64')
        result_df[rank_col_name] = ranked_series

        return result_df

    def rank_multi_columns(
        self,
        df: pd.DataFrame,
        group_cols: Union[str, List[str]],
        value_cols: Union[str, List[str]],
        rank_col_suffix: str = '_rank',
        method: str = 'min',
        ascending: Union[bool, List[bool]] = False,
        na_option: str = 'bottom',
        pct: bool = False,
        dropna: bool = True,
    ) -> pd.DataFrame:
        """
        对多个数值列分别进行分组排名。

        参数:
            df: 输入的 DataFrame
            group_cols: 分组列名或列名列表
            value_cols: 要排名的数值列名或列名列表
            rank_col_suffix: 排名列名后缀，例如 '_rank' -> '销售额_rank'
            method: 并列处理方式
            ascending: 全局升序/降序标志，或对应每个 value_col 的布尔列表
            na_option: 空值处理方式，默认 'bottom'（空值统一排最后）
            pct: 是否以百分比形式显示排名
            dropna: 排名计算时是否排除空值

        返回:
            新增了多个排名列的 DataFrame
        """
        if isinstance(value_cols, str):
            value_cols = [value_cols]

        if isinstance(ascending, bool):
            ascending_list = [ascending] * len(value_cols)
        else:
            ascending_list = list(ascending)
            if len(ascending_list) != len(value_cols):
                raise ValueError(
                    f"ascending 列表长度 ({len(ascending_list)}) 必须 "
                    f"与 value_cols 长度 ({len(value_cols)}) 一致"
                )

        result_df = df.copy()

        for value_col, asc in zip(value_cols, ascending_list):
            rank_col_name = f"{value_col}{rank_col_suffix}"
            result_df = self.rank(
                df=result_df,
                group_cols=group_cols,
                value_col=value_col,
                rank_col_name=rank_col_name,
                method=method,
                ascending=asc,
                na_option=na_option,
                pct=pct,
                dropna=dropna,
            )

        return result_df

    def get_rank_summary(
        self,
        df: pd.DataFrame,
        group_cols: Union[str, List[str]],
        value_col: str,
        method: str = 'min',
        ascending: bool = False,
        top_n: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        获取每个分组内的排名汇总，可选择仅显示 Top N。

        参数:
            df: 输入的 DataFrame
            group_cols: 分组列名或列名列表
            value_col: 要排名的数值列名
            method: 并列处理方式
            ascending: 是否升序排名
            top_n: 仅返回每个分组内的前 N 名，None 表示返回全部

        返回:
            按分组和排名排序的汇总 DataFrame
        """
        ranked_df = self.rank(
            df=df,
            group_cols=group_cols,
            value_col=value_col,
            method=method,
            ascending=ascending,
        )

        if isinstance(group_cols, str):
            group_cols_list = [group_cols]
        else:
            group_cols_list = list(group_cols)

        sort_cols = group_cols_list + ['rank']
        ranked_df = ranked_df.sort_values(sort_cols, na_position='last').reset_index(drop=True)

        if top_n is not None:
            ranked_df = ranked_df[ranked_df['rank'] <= top_n].reset_index(drop=True)

        return ranked_df

    def percentile(
        self,
        df: pd.DataFrame,
        group_cols: Union[str, List[str]],
        value_col: str,
        percentile_col_name: str = 'percentile',
        method: str = 'average',
        ascending: bool = True,
        scale: float = 100.0,
        na_option: str = 'bottom',
        dropna: bool = True,
    ) -> pd.DataFrame:
        """
        计算每个值在其分组内的百分位数。

        参数:
            df: 输入的 DataFrame
            group_cols: 分组列名或列名列表
            value_col: 要计算百分位的数值列名
            percentile_col_name: 输出的百分位列名，默认为 'percentile'
            method: 并列处理方式，可选 'average' | 'min' | 'max' | 'dense' | 'first'
                    影响值相同时的百分位计算方式
            ascending: 是否升序百分位。
                       True=值越大百分位越高（常规百分位，如考试得分率），
                       False=值越小百分位越高（如耗时、成本越低越好）
            scale: 百分位尺度，100.0 表示 0-100 分位，1.0 表示 0-1 的比例
            na_option: 空值处理方式：
                       'bottom' - 空值百分位最低（排最后）
                       'top' - 空值百分位最高（排最前）
                       'keep' - 空值保持 NaN
            dropna: 排名计算时是否排除空值

        返回:
            新增了百分位列的 DataFrame（副本，不修改原数据）
        """
        self._validate_params(df, group_cols, value_col, method, ascending, na_option)

        if scale <= 0:
            raise ValueError("scale 必须大于 0")

        result_df = df.copy()

        if isinstance(group_cols, str):
            group_cols = [group_cols]

        pct_series = result_df.groupby(group_cols, dropna=dropna)[value_col].rank(
            method=method,
            ascending=ascending,
            na_option='keep',
            pct=True,
        )

        if na_option == 'bottom':
            pct_series = pct_series.fillna(0.0)
        elif na_option == 'top':
            pct_series = pct_series.fillna(1.0)

        if scale != 1.0:
            pct_series = pct_series * scale

        result_df[percentile_col_name] = pct_series

        return result_df

    def percentile_multi_columns(
        self,
        df: pd.DataFrame,
        group_cols: Union[str, List[str]],
        value_cols: Union[str, List[str]],
        percentile_col_suffix: str = '_percentile',
        method: str = 'average',
        ascending: Union[bool, List[bool]] = True,
        scale: float = 100.0,
        na_option: str = 'bottom',
        dropna: bool = True,
    ) -> pd.DataFrame:
        """
        对多个数值列分别计算分组百分位。

        参数:
            df: 输入的 DataFrame
            group_cols: 分组列名或列名列表
            value_cols: 要计算百分位的数值列名或列名列表
            percentile_col_suffix: 百分位列名后缀，例如 '_百分位' -> '销售额_百分位'
            method: 并列处理方式
            ascending: 全局升序/降序标志，或对应每个 value_col 的布尔列表
            scale: 百分位尺度，100.0 表示 0-100 分位
            na_option: 空值处理方式，默认 'bottom'
            dropna: 排名计算时是否排除空值

        返回:
            新增了多个百分位列的 DataFrame
        """
        if isinstance(value_cols, str):
            value_cols = [value_cols]

        if isinstance(ascending, bool):
            ascending_list = [ascending] * len(value_cols)
        else:
            ascending_list = list(ascending)
            if len(ascending_list) != len(value_cols):
                raise ValueError(
                    f"ascending 列表长度 ({len(ascending_list)}) 必须 "
                    f"与 value_cols 长度 ({len(value_cols)}) 一致"
                )

        result_df = df.copy()

        for value_col, asc in zip(value_cols, ascending_list):
            percentile_col_name = f"{value_col}{percentile_col_suffix}"
            result_df = self.percentile(
                df=result_df,
                group_cols=group_cols,
                value_col=value_col,
                percentile_col_name=percentile_col_name,
                method=method,
                ascending=asc,
                scale=scale,
                na_option=na_option,
                dropna=dropna,
            )

        return result_df

    def _validate_params(
        self,
        df: pd.DataFrame,
        group_cols: Union[str, List[str]],
        value_col: str,
        method: str,
        ascending: bool,
        na_option: str,
    ) -> None:
        if not isinstance(df, pd.DataFrame):
            raise TypeError("df 必须是 pandas.DataFrame")

        if isinstance(group_cols, str):
            group_cols_list = [group_cols]
        else:
            group_cols_list = list(group_cols)

        missing_cols = [c for c in group_cols_list + [value_col] if c not in df.columns]
        if missing_cols:
            raise ValueError(f"以下列在 DataFrame 中不存在: {missing_cols}")

        if method not in self.VALID_METHODS:
            raise ValueError(
                f"无效的 method 参数 '{method}'，有效值为: {sorted(self.VALID_METHODS)}"
            )

        if pd.api.types.is_numeric_dtype(df[value_col]) is False:
            raise TypeError(f"value_col '{value_col}' 必须是数值类型")

        if na_option not in self.VALID_NA_OPTIONS:
            raise ValueError(
                f"无效的 na_option 参数 '{na_option}'，有效值为: {sorted(self.VALID_NA_OPTIONS)}"
            )
