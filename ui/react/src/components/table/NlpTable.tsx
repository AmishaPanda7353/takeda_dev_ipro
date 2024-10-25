import React, { useEffect, useState } from 'react';
import moment from "moment"
import _ from 'lodash';
import MaterialTable, { Column } from "@material-table/core";
import { AiOutlineLeft, AiOutlineRight } from "react-icons/ai";
import Button from '@mui/material/Button';
import Tooltip from '@mui/material/Tooltip';
import download from "../../assets/images/download_icon.png"
import { ChatHistoryObject } from "../ConversationalFlow/types";


type tableProps = {
    data: any[];
    chatObject: ChatHistoryObject,
    tableError?: string,
    tableShowError?: boolean,
    tableBotResponse: string,
};

function convertToCSV(data: any[]) {
    const columns = Object.keys(data[0]);
    const csvData = [columns.join(',')];


    data.forEach((row) => {
        const values = columns.map((col) => row[col]);

        csvData.push(values.join(','));
    })

    return csvData.join('\n');
}

function downloadCSV(csvData: string, fileName: string) {
    const blob = new Blob([csvData], { type: 'text/csv;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.setAttribute('href', url);
    link.setAttribute('download', fileName);
    link.style.display = 'none';
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
}



function NlpTable({ data, chatObject, tableBotResponse, tableError, tableShowError }: tableProps) {
    const [isButtonDisabled, setIsButtonDisabled] = useState(false);
    const [activeIndex, setActiveIndex] = useState<number>(0);
    const currentData = data[activeIndex] || [];
    const keys = _.keys(currentData[0]) || [];


    const tableColumns: Column<any>[] = keys.map((col: any, index) => ({
        title: col,
        field: col,
        cellStyle: {
            padding: '10px', // Adjust the padding value as per your requirements
        },
        headerStyle: {
            padding: '0px 4px', // Adjust the padding value as per your requirements
        },
    }));

    const pageSize = currentData.length > 5 ? 5 : currentData.length;
    const pageSizeOption = Array.from({ length: pageSize }, (_, i) => i + 1);

    const handleDownload = () => {
        if (!isButtonDisabled) {
            const csvData = convertToCSV(currentData);
            const filename = `table_data_${activeIndex + 1}.csv`;
            downloadCSV(csvData, filename);
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
            {isButtonDisabled ? (
                <Tooltip arrow title="Download is not available for results more than 2 days old" placement="top-start">
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
                    className="absolute right-[16px] top-[26px] z-[1] text-[12px] text-[#18749C] border-[1px] border-[#18749c] rounded-[5px] bg-[#FFFFFF26] px-[8px] py-[6px] flex justify-center items-center"
                >
                    <span className='mr-[4px]'>
                        <img src={download} alt="download-data" />
                    </span>
                    Download Data
                </button>
            )}
            {tableBotResponse && tableBotResponse.length > 0 &&
                <div className='text-sm'>
                    {tableBotResponse}
                </div>
            }
            {tableShowError && tableError && tableError?.length > 0 && (
                <div className='text-sm'>
                    <b>Error:</b>{tableError}
                </div>
            )}

            <div className='mx-[20px] my-[10px] '>
                <MaterialTable
                    title={<div className='text-[1rem]'>{`Table ${activeIndex + 1}`}</div>}
                    columns={tableColumns}
                    data={currentData}
                    options={{
                        headerStyle: {
                            backgroundColor: '#E9EBF0',
                            color: '#1C1C1C',
                            fontWeight: 600,
                            height: "40px",
                            fontSize: "13px",
                        },
                        rowStyle: {
                            fontSize: '13px', // Decrease the font size for rows
                        },
                        pageSize: currentData.length > 5 ? 5 : currentData.length,
                        pageSizeOptions: pageSizeOption,
                        paging: true,
                        search: false,
                        draggable: false,

                    }}
                />
                <div className="flex justify-center items-center py-[16px] text-[13px] ">
                    <AiOutlineLeft data-testid="left-arrow"
                        className={`mr-[10px] text-[16px] text-[#18749C] cursor-pointer ${activeIndex <= 0 ? 'pointer-events-none opacity-[0.6]' : ''
                            }`}
                        onClick={() => setActiveIndex(activeIndex - 1)}
                    />
                    <span data-testid="active-index" className='text-[13px]'>
                        {activeIndex + 1} of {data.length}
                    </span>
                    <AiOutlineRight data-testid="right-arrow"
                        className={`ml-[10px] text-[16px] text-[#18749C] cursor-pointer ${activeIndex >= data.length - 1 ? 'pointer-events-none opacity-[0.6]' : ''
                            }`}
                        onClick={() => setActiveIndex(activeIndex + 1)}
                    />
                </div>
            </div>

        </div>
    );
}


export default NlpTable;
NlpTable.defaultProps = {
    tableShowError: false, // Assuming tableShowError is a boolean prop
    tableError: '',
  };