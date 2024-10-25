import ReactECharts from 'echarts-for-react';

function ScatterChart({ data }: { data: any }) {
  const { xAxisType, yAxisType, xAxisLabel, yAxisLabel, config } = data;
  const option = {
    xAxis: {
      type: xAxisType,
      name: xAxisLabel,
    },
    yAxis: {
      type: yAxisType,
      name: yAxisLabel,
    },
    series: [
      {
        type: 'scatter',
        data: config[0]?.data.map((item: any) => [item.x, item.y]) || [],
      },
    ],
  };

  return <ReactECharts option={option} />;
};

export default ScatterChart;