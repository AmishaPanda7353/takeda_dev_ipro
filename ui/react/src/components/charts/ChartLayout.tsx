import React, { useState } from 'react'
import Charts from './Charts';

function ChartLayout() {
  const [chartType, setChartType] = useState('bar');

  const handleChartTypeChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
    setChartType(e.target.value);
  };

  const chartData =
  {
    xAxisType: 'category',
    yAxisType: 'value',
    xAxisLabel: 'Categories',
    yAxisLabel: 'Values',
    type: 'scatter',
    config: [
      {
        type: 'bar',
        data: [
          { x: 'A', y: 10 },
          { x: 'B', y: 20 },
          { x: 'C', y: 30 },
          { x: 'D', y: 40 },
          { x: 'E', y: 60 },
          { x: 'F', y: 10 },
          { x: 'G', y: 35 },
          { x: 'H', y: 90 },
        ],
      },
    ]
  };
  const mutilpleChartData =
  {
    xAxisType: 'category',
    yAxisType: 'value',
    xAxisLabel: 'Categories',
    yAxisLabel: 'Values',
    type: 'line',
    config: [
      {
        type: 'bar',
        data: [
          { x: 'A', y: 10 },
          { x: 'B', y: 20 },
          { x: 'C', y: 30 },
          { x: 'D', y: 40 },
          { x: 'E', y: 60 },
          { x: 'F', y: 10 },
          { x: 'G', y: 35 },
          { x: 'H', y: 90 },
        ],
      },
      {
        type: 'line',
        data: [
          { x: 'A', y: 10 },
          { x: 'B', y: 20 },
          { x: 'C', y: 30 },
          { x: 'D', y: 40 },
          { x: 'E', y: 60 },
          { x: 'F', y: 10 },
          { x: 'G', y: 35 },
          { x: 'H', y: 90 },
        ],
      },
    ]
  };

  return (
    <div>
      <h1>SingleChart</h1>
      <Charts chartData={chartData} />
      <h1>MultpleChart</h1>
      <Charts chartData={mutilpleChartData} />
    </div>
  );
}

export default ChartLayout