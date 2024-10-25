/* eslint-disable security/detect-object-injection */
/* eslint-disable camelcase */
import React, { useEffect, useState } from "react";
import ResetChat from "components/ResetChatModal";
import { useAppSelector, useAppDispatch } from "store";
import moment from "moment";
import client from "client";
import { ENDPOINTS } from "constants/constant";
import {
  setActiveChatDetails,
  setActiveReportChatId,
  setIsOldReport,
  setOutletLoading,
  setUserChatDetails,
} from "store/slice/userSlice";
import {
  ReportObject,
  chatHistoryAPIResponse,
  chatdetails,
} from "components/ConversationalFlow/types";
import { getUserChatDetailsReponse } from "constants/common";
import { SidebarProps } from "./types";
import tigersidebarlogo from "../../assets/images/footer-tigerlogo.png";
import reset from "../../assets/images/restart_alt.png";
import documentIcon from "../../assets/images/document_scanner.png";

function Sidebar({ initialOpenState }: SidebarProps) {
  const {
    user_id,
    activeReportChatId,
    activeChatId,
    chat_details,
    activeChatIndex,
    isOldReport,
    selectedDomain,
  } = useAppSelector((s) => s.user);
  const dispatch = useAppDispatch();
  const { report_list } = useAppSelector((s) => s.user);
  const [open, setOpen] = useState<boolean>(initialOpenState);
  const [resetModalShow, setResetModalShow] = useState<boolean>(false);

  const getInitialChatHistory = async () => {
    dispatch(setOutletLoading(true));
    const payload: any = { user_id, data_name: selectedDomain };
    const response = await getUserChatDetailsReponse(payload);
    dispatch(setUserChatDetails(response.data));
    dispatch(setOutletLoading(false));
  };

  // useEffect(() => {
  //   getInitialChatHistory();
  // }, []);

  const handleReportClick = async (report: ReportObject) => {
    dispatch(setOutletLoading(true));
    dispatch(setActiveReportChatId(report.chat_id));
    dispatch(
      setIsOldReport(report.chat_id.toString() !== activeChatId.toString())
    );
    if (report.chat_id.toString() !== activeChatId.toString()) {
      const response: chatHistoryAPIResponse = await client.post(
        ENDPOINTS.chatHistory,
        {
          chat_ids: [report.chat_id],
          domains: selectedDomain,
        }
      );
      const chatDets: chatdetails = {
        chat_id: report.chat_id,
        context_id: 0,
        user_id,
        chat_name: "",
        data_name: "",
        is_active: false,
        chat_history: response.data,
      };
      dispatch(setActiveChatDetails([chatDets]));
    } else {
      dispatch(setActiveChatDetails([chat_details[activeChatIndex]]));
    }
    dispatch(setOutletLoading(false));
    // window.open(report.report_url, '_blank');
  };

  function formatDateTime(reportNameList: any) {
    // Check if the input is empty or falsy
    if (!reportNameList) {
      return reportNameList; // Return the empty or falsy input as-is
    }

    // Attempt to parse and format the UTC time
    const utcTimeString = reportNameList;
    const localTime = moment.utc(utcTimeString, "DD MMM YYYY_HH:mm").local();

    // Check if the localTime is valid
    if (!localTime.isValid()) {
      return reportNameList; // Return the input as-is if it couldn't be parsed
    }

    // Format the local time and return it
    const formattedLocalTime = localTime.format("DD MMM YYYY HH:mm");
    return formattedLocalTime;
  }

  return (
    <div className="bg-[#020C17] min-h-screen w-48 text-gray-100">
      <div
        data-testid="sidebar"
        className={` ${
          isOldReport ? "opacity-50" : ""
        } py-3 m-auto my-3 flex justify-center items-center bg-[#18749C4D] rounded-[5px] w-[130px] border-[1px] border-solid border-[#18749C] cursor-pointer`}
      >
        <img className="w-[12px] mr-[5px]" src={reset} alt="reset-chat" />
        <button
          disabled={isOldReport}
          onClick={() => {
            setResetModalShow(true);
          }}
          type="button"
          className=""
        >
          Reset Chat
        </button>
      </div>

      {resetModalShow && (
        <ResetChat
          resetModalShow={resetModalShow}
          setResetModalShow={setResetModalShow}
        />
      )}

      {report_list.length > 0 && activeReportChatId && (
        <div
          data-testid="HandleReportDiv"
          className={`${
            activeReportChatId === report_list[0].chat_id
              ? "bg-[rgba(255,255,255,0.3)]"
              : ""
          } ml-[20px] flex flex-col cursor-pointer`}
          onClick={() => handleReportClick(report_list[0])}
        >
          <div className="flex m-[10px] items-center">
            <div>{report_list[0]?.report_name}</div>
          </div>
        </div>
      )}

      <div className="ml-[20px] mt-[12px] text-sm Poppins">
        Reports
        <div
          data-testid="ReportDiv"
          className="h-[calc(100vh-210px)] scrollbar-thin scrollbar-thumb-gray-400 scrollbar-track-bg-[#020C17]"
        >
          {report_list.length > 1 ? (
            report_list
              .slice(1)
              .reverse()
              .map((val: ReportObject) => (
                <div
                  data-testid="HandleReportDiv"
                  className={`${
                    activeReportChatId === val.chat_id
                      ? "bg-[rgba(255,255,255,0.3)]"
                      : ""
                  } flex flex-col cursor-pointer`}
                  key={val?.report_name}
                  onClick={() => handleReportClick(val)}
                >
                  <div className="flex m-[10px] items-center">
                    <img
                      className="w-[10px] h-[11px] mr-[5px]"
                      src={documentIcon}
                      alt="document-scanner"
                    />
                    <div className="text-[13px]">
                      {formatDateTime(val?.report_name)}
                    </div>
                  </div>
                </div>
              ))
          ) : (
            <div className="mt-[5px] text-sm">No report created </div>
          )}
        </div>
      </div>
      <div className="mt-4 flex flex-col gap-4 relative  h-[calc(100vh-160px)]">
        {open ? (
          <div className="flex fixed bottom-4 items-center">
            <div className="flex text-left text-[#9B9898] text-sm ml-[10px] opacity-1 font-poppins">
              <div>Powered</div>
              <div className="ml-[5px]">by</div>
            </div>
            <img
              className="ml-[10px] w-[66px]"
              src={tigersidebarlogo}
              alt="tiger"
            />
          </div>
        ) : (
          <div className="flex absolute bottom-2 w-[66px]">
            <img className="w-[66px]" src={tigersidebarlogo} alt="tiger" />
          </div>
        )}
      </div>
    </div>
  );
}

export default Sidebar;
