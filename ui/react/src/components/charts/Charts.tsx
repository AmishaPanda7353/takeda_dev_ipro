import React, { useState } from 'react';
import ReactECharts from 'echarts-for-react';
import BarChart from './chartsGroup/BarChart';
import LineChart from './chartsGroup/LineChart';
import PieChart from './chartsGroup/PieChart';
import ScatterChart from './chartsGroup/ScatterChart';
import HistogramChart from './chartsGroup/HistogramChart';
import StackedChart from './chartsGroup/StackedChart';

function Charts({ chartData }: { chartData: any }) {

  const [chartType, setChartType] = useState(chartData.type)

  const handleChartTypeChange = (e: any) => {
    setChartType(e.target.value)
  }

  const renderChart = () => {
    switch (chartType) {
      case 'bar':
        return <BarChart data={chartData} />;
      case 'line':
        return <LineChart data={chartData} />;
      case 'pie':
        return <PieChart data={chartData} />;
      case 'scatter':
        return <ScatterChart data={chartData} />;
      case 'histogram':
        return <HistogramChart data={chartData} />;
      case 'stacked':
        return <StackedChart data={chartData} />;
      default:
        return null;
    }
  }

  return (
    <div>
      <div className='mr-[10px] flex justify-end'>
        <select className='rounded-[6px] ' onChange={handleChartTypeChange} value={chartType}>
          <option value="bar">Bar Chart</option>
          <option value="line">Line Chart</option>
          <option value="pie"> Pie Chart</option>
          <option value="scatter">Scatter chart</option>
        </select>
      </div>
      {renderChart()}
    </div>
  )

};

export default Charts;
