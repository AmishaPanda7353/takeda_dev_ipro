import ReactECharts from 'echarts-for-react';

function StackedChart({ data }: { data: any }) {
  const { config, xAxisType, yAxisType, xAxisLabel, yAxisLabel, } = data;
  const seriesData = config.map((series: any) => ({
    name: series.name,
    type: 'line',
    stack: 'stacked',
    data: series.data.map((item: any) => ({
      name: item.x,
      value: item.y,
    })),
  }));

  const options = {
    xAxis: {
      type: xAxisType,
      data: ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'],
    },
    yAxis: {
      type: yAxisType,
      name: yAxisLabel,
    },
    series: seriesData,
  };

  return <ReactECharts option={options} />;
};
export default StackedChart;