import ReactECharts from 'echarts-for-react';

function PieChart({ data }: { data: any }) {
  const { config } = data;
  const option = {
    series: [
      {
        type: 'pie',
        data: config[0]?.data.map((item: any) => ({
          name: item.x,
          value: item.y,
        })) || [],
      },
    ],
  };

  return <ReactECharts option={option} />;
};
export default PieChart;