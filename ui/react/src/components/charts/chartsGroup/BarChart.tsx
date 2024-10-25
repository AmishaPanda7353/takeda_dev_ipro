import React from 'react';
import ReactECharts from 'echarts-for-react';

function BarChart({ data }: { data: any }) {
  const { xAxisType, yAxisType, xAxisLabel, yAxisLabel, config } = data;

  const option = {
    xAxis: {
      type: xAxisType,
      name: xAxisLabel,
      data: config[0].data.map((item: any) => item.x),
    },
    yAxis: {
      type: yAxisType,
      name: yAxisLabel,
    },
    series: [
      {
        type: 'bar',
        data: config[0].data.map((item: any) => item.y),
      },
    ],
  };

  return <ReactECharts option={option} />;
}

export default BarChart;
