import ReactECharts from 'echarts-for-react';

function HistogramChart({ data }: { data: any }) {
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
        type: 'histogram',
        data: config[0]?.data.map((item: any) => ({
          name: item.x,
          value: item.y,
        })) || [],
      },
    ],
  };

  return <ReactECharts option={option} />;
};
export default HistogramChart