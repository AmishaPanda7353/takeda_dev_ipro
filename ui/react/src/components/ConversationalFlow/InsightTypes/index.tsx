/* eslint-disable security/detect-object-injection */
/* eslint-disable react/no-array-index-key */
/* eslint-disable import/extensions */
import React, { useState, useEffect } from 'react';
import moment from "moment"
import NlpTable from 'components/table/NlpTable';
import Accordion from '@mui/material/Accordion';
import ToggleButton from '@mui/material/ToggleButton';
import ToggleButtonGroup from '@mui/material/ToggleButtonGroup';
import AccordionSummary from '@mui/material/AccordionSummary';
import ExpandMoreIcon from '@mui/icons-material/ExpandMore';
import AccordionDetails from '@mui/material/AccordionDetails';
import { Tooltip } from '@mui/material';
import Insight from '../Insight';
import DottedLine from '../DottoedLine/DottedLine';
import Query from '../Query';
import download from "../../../assets/images/download_icon.png"
import regenerate from "../../../assets/images/regenerate.png"
import PlotlyComponent from './PlotlyComponent';
import InsightNumber from '../InsightNumber/InsightNumber';
import BotImage from "../../../assets/images/BotImage.svg"
import UserAnswerFeedback from '../UserAnswerFeedback';
import { ChatHistoryObject } from "../types";


function InsightType({ data, chatObject }: { data: any, chatObject: ChatHistoryObject }) {
  const [isButtonDisabled, setIsButtonDisabled] = useState(false);
  const [showPlotly, setShowPlotly] = useState(true); // Default is Graph

  const handleToggleButtonChange = (event: React.MouseEvent<HTMLElement>, newShowPlotly: any) => {
    if (newShowPlotly !== null) {
      setShowPlotly(newShowPlotly === 'Graph');
    }
  }

  // disabled download data button after two days
  useEffect(() => {
    const reportTime = moment(chatObject.created_time);
    const currentTime = moment();
    const twoDaysAgo = currentTime.subtract(2, 'days');
    if (reportTime.isBefore(twoDaysAgo)) {
      setIsButtonDisabled(true);
    }
  }, []);

  // To test the validJSON or not 
  function isValidJson(str: string) {
    try {
      // Replace 'None' with 'null' before parsing
      const modifiedStr = str.replace(/None/g, 'null');
      JSON.parse(modifiedStr);
      return true;
    } catch (e:any) {
      console.error('Invalid JSON:', e.message);
      return false;
    }
  }


  // ordering insightType catergory 
  const customSort = ({ dataIp, sortBy, sortField }: { dataIp: any, sortBy: any, sortField: any }) => {
    const sortByObject = sortBy.reduce((obj: any, item: any, index: any) => ({
      ...obj,
      [item]: index,
    }), {});
    return dataIp.slice().sort(
      (a: any, b: any) => sortByObject[a[sortField]] - sortByObject[b[sortField]]
    );
  };
  const orderData = customSort(
    {
      dataIp: data,
      sortBy: ["scalar", "table_selector", "table", "chart", "multi_chart", "summary", "sql_query"],
      sortField: "insight_type"
    }
  );

  return (
    <div>
      <div className=' relative bg-[#FFFFFF]  min-h-[144px] rounded-[10px] opacity-[1px] m-[20px]'>
        <div className='flex justify-between items-center pl-[10px] pt-[30px] mb-[40px] pr-[28px]'>
          <div className='flex justify-start items-center pl-[50px]'>
            <div className='flex justify-center items-center bg-black rounded-full h-[26px] w-[26px]'>
              <img className='w-[20px]' src={BotImage} alt="chatbotICon" />
            </div>
            <div className='pl-[10px] pr-[10px] text-[14px] font-poppins text-[#1c1c1c]'>
              {chatObject?.data?.completion_response}
            </div>
          </div>
        </div>

        <div className='border-[#DEE3EA] border-2 rounded-[5px] p-[12px] w-[90%] ml-[8%] md:ml-[12%] md:w-[82%] lg:ml-[8%]'>
          {orderData.map((val: any, idx: any) =>
          (
            <div className='' key={idx}>
              {
                val.insight_type === "summary" ?
                  <div className='m-[20px]'>
                    <Insight
                      insightText={val?.content}
                      errorSummary={val?.error}
                      showErrorSummary={val?.showError}
                      botResponseSummary={val?.bot_response}
                    />
                  </div>
                  :
                  val.insight_type === "sql_query" ?
                    <Accordion>
                      <AccordionSummary expandIcon={<ExpandMoreIcon />} aria-controls={`panel-${idx}-content`} id={`panel-${idx}-header`}>
                        {/* Render Accordion Header */}
                        <div className='text-[14px] font-semibold'>
                          SQL Query
                        </div>
                      </AccordionSummary>
                      <AccordionDetails>
                        {/* Render the Query component */}
                        <Query query={val?.content} error={val?.error} showError={val?.showError} botResponseQuery={val?.bot_response} />
                      </AccordionDetails>
                    </Accordion>
                    :
                    val.insight_type === "chart" || val.insight_type === "multi_chart" && val?.content && typeof (val?.content) !== "string" ?
                      <div>
                        <div>
                          <ToggleButtonGroup
                            color='primary'
                            value={showPlotly ? 'Graph' : 'Table'}
                            exclusive
                            onChange={handleToggleButtonChange}
                            aria-label="plotly-or-table"
                          >
                            <ToggleButton
                              value="Graph"
                              aria-label="plotly"
                              className='rounded-[6px] p-[9px]'
                            // className={`rounded-[6px] p-[9px] ${!showPlotly ? 'bg-[#4e555d] text-white' : 'bg-[#000000] text-black'}`}
                            >
                              Graph
                            </ToggleButton>
                            <ToggleButton
                              value="Table"
                              aria-label="table"
                              className='rounded-[6px] p-[9px]'
                            //  className={`rounded-[6px] p-[9px] ${!showPlotly ? 'bg-[#4e555d] text-white' : 'bg-[#000000] text-black'}`}
                            >
                              Table
                            </ToggleButton>
                          </ToggleButtonGroup>
                        </div>
                        <div className="m-[20px]">
                          <div className={`mt-4 ${showPlotly ? 'Graph flex' : ''} `} >
                            {showPlotly ? (
                              <PlotlyComponent
                                chatObject={chatObject}
                                graphData={val}
                                chartJson={val?.content}
                                error={val?.error}
                                showError={val?.showError}
                                botResponse={val?.bot_response}
                              />
                            ) : (
                              <NlpTable
                                data={[JSON.parse(val?.table[0])]}
                                chatObject={chatObject}
                                tableBotResponse={val?.bot_response} />
                            )}
                          </div>
                        </div>
                      </div>
                      : val.insight_type === "table" ?
                      <div className=''>
                      {(() => {
                        try {
                          const jsonData = JSON.parse(val?.content[0]);
                          return <NlpTable data={[jsonData]} chatObject={chatObject} tableError={val?.error} tableShowError={val?.showError} tableBotResponse={val?.bot_response} />;
                        } catch (error) {
                          console.error("Error parsing JSON:", error);
                          return <NlpTable data={[]} chatObject={chatObject} tableError={val?.error} tableShowError={val?.showError} tableBotResponse={val?.bot_response} />;
                        }
                      })()}
                    </div>
                        : val.insight_type === "scalar" ? <div className='m-[20px]'> <InsightNumber num={val?.content} /></div> : " "
              }
              {
                data.length - 1 !== idx ? <DottedLine /> : ""
              }
            </div>
          ))}
        </div>
        <div className='flex items-center justify-between ml-[98px] mr-[10%] mt-[10px] text-[14px] text-[#18749C] pb-6'
        // onClick={() => console.log("Regenerate Response Created")}
        >
          <Tooltip placement='top' title="Regenerate response (Work in Progress - Will be functional in the next release)">
            <div className='flex items-center opacity-[0.5]'>
              <img className='mr-[6px]' src={regenerate} alt="refresh" />
              <div>Regenerate Response</div>
            </div>
          </Tooltip>
          <div>
            <UserAnswerFeedback
              key={String(chatObject.feedback)}
              answerId={chatObject.answer_id?.toString()}
              defaultLike={chatObject.feedback !== null && chatObject.feedback === true}
              defaultDislike={chatObject.feedback !== null && chatObject.feedback === false}
              contextId={chatObject.context_id} />
          </div>
        </div>
      </div>
    </div >
  )
}

export default InsightType