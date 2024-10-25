/* eslint-disable camelcase */
import React, { useState } from 'react'
import _ from "lodash";
import { Modal } from "@mui/material";
import { useAppSelector, useAppDispatch } from "store";
import { chatdetails } from 'components/ConversationalFlow/types';
import { setOutletLoading, setUserChatDetails } from 'store/slice/userSlice';
import client from 'client';
import { ENDPOINTS } from 'constants/constant';
import { getUserChatDetailsReponse } from 'constants/common';
import ValidDocumentScanner from '../../assets/images/ValidDocumentScanner.png';


function GenerateReport() {
  const dispatch = useAppDispatch();
  const { isNewUser, activeChatId, user_id, activeChatDetails, isOldReport, report_list, selectedDomain } = useAppSelector((s) => s.user);

  const [modalShow, setModalShow] = useState<boolean>(false);
  const activeChat: chatdetails = activeChatDetails[0];

  const generateReport = async () => {
    if (isOldReport) {
      const currentReport = _.filter(report_list, (i) => i.chat_id === activeChat.chat_id)[0];
      window.open(currentReport.report_url);
      setModalShow(false);
    } else {
      dispatch(setOutletLoading(true));
      const reportResponse = await client.post(ENDPOINTS.generateReport,
        {
          chat_id: activeChatId.toString(),
          user_id: user_id.toString(),
          data_name: selectedDomain
        });
      console.log(reportResponse, "url")
      window.open(reportResponse?.data?.report?.report_url);
      // console.log(reportResponse?.data?.report_url, "log")
      const payload: any = { user_id, data_name: selectedDomain };
      const response = await getUserChatDetailsReponse(payload);
      dispatch(setUserChatDetails(response.data));
      dispatch(setOutletLoading(false));
      setModalShow(false)
    }
  }

  return (
    <>
      <div className="bg-white w-100 flex justify-center p-2">
        <button type='button' className="h-[32px] w-[134px] flex justify-center align-center items-center border border-solid border-[#18749C] rounded-[5px] opacity-100 disabled:opacity-[0.5] bg-white"
          onClick={() => setModalShow(true)}
          disabled={isOldReport ? false : isNewUser}
        >
          <img src={ValidDocumentScanner} alt="Document Scanner" />
          <span className='font-poppins ml-1 text-[12px] text-[#18749C] font-regular'>{`${isOldReport ? "Download Report" : "Generate Report"}`} </span>
        </button>
      </div>

      <div>
        {
          modalShow && (
            <Modal open={modalShow} onClose={() => setModalShow(false)} className="">
              <div className="bg-white absolute  rounded-[10px] top-1/2 left-1/2 transform -translate-x-2/4 -translate-y-2/4 p-[20px] w-[600px]  outline-none overflow:scroll">
                <div className='font-bold'>Reports</div>
                <div className="text-[#1C1C1C] text-left text-[14px] py-4">Below are the list of questions to be downloaded.</div>
                <div className=' h-64 pl-[8px] pr-[16px] scrollbar-thin scrollbar-thumb-gray-400 scrollbar-track-white'>
                  {
                    activeChatDetails.length > 0 && _.size(activeChat?.chat_history) > 0 ?
                      <>
                        {
                          _.map(activeChat.chat_history, (history, key) => (
                            <div key={`${key}-ppgenrep`} className='px-[8px] py-[5px] mb-3 border-[1px] border-solid border-[#DEE3EA]'>
                              {
                                _.map(history.map((val, idx) => (
                                  <div className='py-2'>Q{idx + 1}: {val.question}</div>
                                )))
                              }
                            </div>
                          ))
                        }
                      </>
                      : "No reports available!"
                  }
                </div>
                < div className="flex justify-end gap-5 mt-[19px] " >
                  <button type="button"
                    className="font-poppins cursor-pointer rounded bg-white text-[#18749C] h-[40px] w-[104px] opacity-[1px] hover:border-[#18749C] hover:border-[1px]"
                    onClick={() => { setModalShow(false) }}
                  >
                    Cancel
                  </button>
                  <button type="button"
                    className="font-poppins rounded text-white bg-[#18749C] h-[40px] w-[104px] opacity-[1px]"
                    onClick={generateReport}
                  >
                    Download
                  </button>
                </div>
              </div>
            </Modal >
          )
        }
      </div>
    </>
  )
}

export default GenerateReport