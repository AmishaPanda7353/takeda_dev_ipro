import React, { useEffect, useState } from "react";
import Plot from "react-plotly.js";
import _ from 'lodash';
import moment from "moment";
import { Tooltip } from "@mui/material";
import { IoChevronForward } from 'react-icons/io5'
import { IoIosArrowBack } from 'react-icons/io'
import download from "../../../assets/images/download_icon.png";
import { ChatHistoryObject } from "../types";

export default function PlotlyComponent({ graphData, chatObject, chartJson, error, showError, botResponse }: { graphData: any, chatObject: ChatHistoryObject, chartJson: any, error: string, showError: boolean, botResponse: string }) {

  const [clonedChartJson, setClonedChartJson] = useState(_.cloneDeep(chartJson));
  const [isButtonDisabled, setIsButtonDisabled] = useState(false);
  const [currentPlotIndex, setCurrentPlotIndex] = useState(0);

  const handleDownload = () => {
    const csvData = graphData.table[0]; // Extracting the CSV data from the array
    const parsedData = JSON.parse(csvData); // Parsing the JSON data

    const headers = Object.keys(parsedData[0]).join(','); // Extracting headers

    const rows = parsedData.map((obj: any) =>
      Object.values(obj).join(',')
    ).join('\n'); // Extracting rows

    const csvText = `${headers}\n${rows}`; // Combining headers and rows

    const filename = `plot_data.csv`;
    const blob = new Blob([csvText], { type: 'text/csv;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.setAttribute('href', url);
    link.setAttribute('download', filename);
    link.style.display = 'none';
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  }

  const handleNext = () => {
    if (currentPlotIndex < clonedChartJson.length - 1) {
      setCurrentPlotIndex(currentPlotIndex + 1);
    }
  }

  const handlePrevious = () => {
    if (currentPlotIndex > 0) {
      setCurrentPlotIndex(currentPlotIndex - 1);
    }
  }

  useEffect(() => {
    const reportTime = moment(chatObject.created_time);
    const currentTime = moment();
    const twoDaysAgo = currentTime.subtract(2, 'days');
    if (reportTime.isBefore(twoDaysAgo)) {
      setIsButtonDisabled(true);
    }
  }, []);

  return (
    <div >
      <div className="flex flex-col items-center">
        {isButtonDisabled ? (
          <Tooltip
            arrow
            title="Download is not available for results more than 2 days old"
            placement="top-start"
          >
            <button
              type="button"
              disabled={isButtonDisabled}
              onClick={handleDownload}
              className={` ${isButtonDisabled ? " opacity-[0.5]" : ""} absolute right-[16px] top-[26px] z-[1]  text-[12px] text-[#18749C] border-[1px] border-[#18749c] rounded-[5px] bg-[#FFFFFF26] px-[8px] py-[6px]  flex justify-center items-center`}
            >
              <span className='mr-[4px]'>
                <img src={download} alt="download-data" />
              </span>
              Download Data
            </button>
          </Tooltip>
        ) : (
          <button
            type="button"
            disabled={isButtonDisabled}
            onClick={handleDownload}
            className="absolute right-[16px] top-[26px] z-[1]  text-[12px] text-[#18749C] border-[1px] border-[#18749c] rounded-[5px] bg-[#FFFFFF26] px-[8px] py-[6px]  flex justify-center items-center"
          >
            <span className='mr-[4px]'>
              <img src={download} alt="download-data" />
            </span>
            Download Data
          </button>
        )}

        {botResponse && botResponse.length > 0 && (
          <div className="text-sm">
            {botResponse}
          </div>
        )}
        {
          showError && error && error.length > 0 && (
            <div className="text-sm">
              <b>Error:</b> {error}
            </div>
          )
        }

        <Plot data={clonedChartJson[currentPlotIndex]?.data} layout={clonedChartJson[currentPlotIndex]?.layout} config={{ displaylogo: false, scrollZoom: true }} />
        {
          clonedChartJson.length > 1 &&
          <div className="flex justify-center items-center text-[1.2rem] ">
            <IoIosArrowBack onClick={handlePrevious} className={`border-2 border-blue-500 rounded-[14px] text-[24px] mr-10 p-[2px] icon ${currentPlotIndex === 0 ? 'disabled opacity-[0.5]' : ''}`} />
            <span className="index-display ">
              {currentPlotIndex + 1}/{clonedChartJson.length}
            </span>
            <IoChevronForward onClick={handleNext} className={`border-2 border-blue-500 rounded-[14px] text-[24px] ml-10 p-[2px] icon ${currentPlotIndex === clonedChartJson.length - 1 ? 'disabled opacity-[0.5]' : ''}`} />
          </div>}

      </div>
    </div>
  );
}
