/* eslint-disable react/no-array-index-key */
/* eslint-disable security/detect-object-injection */
/* eslint-disable camelcase */
import React, { useState } from "react";
import { useAppSelector, useAppDispatch } from "store";
import { TextareaAutosize, Tooltip } from "@mui/material";
import AddIcon from "@mui/icons-material/Add";
import {
  ChatHistoryObject,
  mergeTrackDataObject,
  newContextAPIResponse,
} from "components/ConversationalFlow/types";
import client from "client";
import { ENDPOINTS } from "constants/constant";
import {
  addUserChatDetails,
  setOutletLoading,
  setUserChatDetails,
  addUserChatQuestionDetails,
  addNewContextId,
  setIsFetchingQuesResponse,
  setCombineTrackData,
  setFalseCount,
  updateUserChatDetails,
} from "store/slice/userSlice";
import { getUserChatDetailsReponse } from "constants/common";
import { filter } from "lodash";
import InvalidSend from "../../assets/images/InvalidSend.png";
import ValidSend from "../../assets/images/ValidSend.png";

function ChatInput({
  scrollToBottom,
  isLoading,
}: {
  scrollToBottom: any;
  isLoading: boolean;
}) {
  const dispatch = useAppDispatch();
  const {
    isNewUser,
    activeContextId,
    activeChatId,
    user_id,
    isOldReport,
    selectedDomain,
  } = useAppSelector((s) => s.user);
  const [chatQuestion, setChatQuestion] = useState<string>("");
  const [error, setError] = useState<string>(""); // State to track error message

  const handleChange = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    setChatQuestion(e.target.value);
  };

  const sendQuestion = async () => {
    const ques: string = chatQuestion;

    const tempHist: ChatHistoryObject = {
      question_id: 0,
      context_id: activeContextId,
      feedback: null,
      question: chatQuestion,
      answer_id: 0,
      type: "",
      data: null,
      created_time: new Date().toISOString(),
      isWaiting: true,
    };
    setChatQuestion("");
    dispatch(addUserChatQuestionDetails(tempHist));
    dispatch(setIsFetchingQuesResponse(true)); // Set loading state to true
    setError(""); // Reset error message

    try {
      const responsePreTrack: any = await client.post(ENDPOINTS.preTrack, {
        context_id: activeContextId.toString(),
        user_id: user_id.toString(),
        question: ques,
      });

      const resPreTrackData: any = responsePreTrack.data;
      // console.log("response", resPreTrackData);

      const falseCount = filter(
        resPreTrackData,
        (value) => value === false
      ).length;

      // console.log("False Count", falseCount);
      dispatch(setFalseCount(falseCount));

      const { question_id, answer_id } = resPreTrackData[0];
      const { track_1, track_2, track_3 } = resPreTrackData[3];
      const qId = question_id;
      const aId = answer_id;

      let dataArr: mergeTrackDataObject[] = [];
      let chatHist: ChatHistoryObject = {
        question_id: 0,
        context_id: activeContextId,
        feedback: null,
        question: chatQuestion,
        answer_id: 0,
        type: "",
        data: null,
        created_time: new Date().toISOString(),
        isWaiting: true,
      };

      let postTrack1Resp: any;
      let postTrack2Resp: any;
      let postTrack3Resp: any;
      let questionPath: string = "";

      try {
        // Running track 1 API
        const track1: boolean = track_1;
        if (!track1 || null) {
          const responseTrack1 = await client.post(ENDPOINTS.track1, {
            question: ques,
            additional_context: "",
            language: "english",
            // domain_name: selectedDomain,
            user_id: user_id.toString(),
            question_id: qId,
            answer_id: aId,
            question_path: ""
          });

          const respTrack1Data: any = responseTrack1?.data;
          // console.log("Track 1 Response:", respTrack1Data);

          postTrack1Resp = respTrack1Data;

          const data_track1 = respTrack1Data?.content?.data[0];
          questionPath = respTrack1Data?.question_path
          if (data_track1) {
            // dataArr.push(data_track1);
            dataArr = [...dataArr, { ...data_track1 }];
            // console.log("Updated after", dataArr);
            dispatch(setCombineTrackData(dataArr));
          }

          const chatHistObj: ChatHistoryObject = {
            question_id: respTrack1Data?.question_id,
            context_id: activeContextId,
            feedback: null,
            question: ques,
            answer_id: respTrack1Data?.answer_id,
            type: respTrack1Data?.category,
            data: respTrack1Data?.content,
            created_time: respTrack1Data?.content?.created_time,
            error: respTrack1Data?.content?.error,
            status: respTrack1Data?.content?.status,
            isWaiting: false,
          };

          chatHist = {
            ...chatHistObj,
            data: { ...chatHistObj?.data, data: dataArr },
          };
          // console.log("Track 1 =====", chatHist);

          // dispatch(updateUserChatDetails(chatHistObj));
          // scrollToBottom();
        }

        // Running track 2 API
        const track2: boolean = track_2;
        if (!track2) {
          const responseTrack2 = await client.post(ENDPOINTS.track2, {
            question: ques,
            additional_context: "",
            language: "english",
            // domain_name: selectedDomain,
            user_id: user_id.toString(),
            question_id: qId,
            answer_id: aId,
            question_path: questionPath
          });

          const respTrack2Data: any = responseTrack2?.data;
          // console.log("Track 2 Response:", respTrack2Data);

          postTrack2Resp = respTrack2Data;

          const track2Data = {
            ...respTrack2Data,
            content: {
              ...respTrack2Data?.content,
              data: filter(
                respTrack2Data?.content?.data,
                (ele) =>
                  ele.insight_type === "chart" || ele.insight_type === "table"
              ),
            },
          };
          // console.log("Modified Track 2 Data", track2Data);

          const data_track2 = track2Data?.content?.data[0];

          if (data_track2) {
            // dataArr.push(data_track2);
            dataArr = [...dataArr, { ...data_track2 }];
            // console.log("Updated after", dataArr);

            dispatch(setCombineTrackData(dataArr));
          }

          const chatHistObj: ChatHistoryObject = {
            question_id: track2Data?.question_id,
            context_id: activeContextId,
            feedback: null,
            question: ques,
            answer_id: track2Data?.answer_id,
            type: track2Data?.category,
            data: track2Data?.content,
            created_time: track2Data?.content?.created_time,
            error: track2Data?.content?.error,
            status: track2Data?.content?.status,
            isWaiting: false,
          };

          chatHist = {
            ...chatHistObj,
            data: { ...chatHistObj?.data, data: dataArr },
          };
          // console.log("Track 2 =====", chatHist);

          // dispatch(updateUserChatDetails(chatHistObj));
          // scrollToBottom();
        }

        // Running track 3 API
        const track3: boolean = track_3;

        if (!track3) {
          const responseTrack3 = await client.post(ENDPOINTS.track3, {
            question: ques,
            additional_context: "",
            language: "english",
            // domain_name: selectedDomain,
            user_id: user_id.toString(),
            question_id: qId,
            answer_id: aId,
            question_path: questionPath
          });

          const respTrack3Data: any = responseTrack3?.data;
          // console.log("Track 3 Response:", respTrack3Data);

          postTrack3Resp = respTrack3Data;

          const track3Data = {
            ...respTrack3Data,
            content: {
              ...respTrack3Data?.content,
              data: filter(
                respTrack3Data?.content?.data,
                (ele) => ele.insight_type === "summary"
              ),
            },
          };
          // console.log("Modified Track 3 Data", track3Data);

          const data_track3 = track3Data?.content?.data[0];

          if (data_track3) {
            // dataArr.push(data_track3);
            dataArr = [...dataArr, { ...data_track3 }];
            // console.log("Updated after", dataArr);

            dispatch(setCombineTrackData(dataArr));
          }

          const chatHistObj: ChatHistoryObject = {
            question_id: track3Data?.question_id,
            context_id: activeContextId,
            feedback: null,
            question: ques,
            answer_id: track3Data?.answer_id,
            type: track3Data?.category,
            data: track3Data?.content,
            created_time: track3Data?.content?.created_time,
            error: track3Data?.content?.error,
            status: track3Data?.content?.status,
            isWaiting: false,
          };

          chatHist = {
            ...chatHistObj,
            data: { ...chatHistObj?.data, data: dataArr },
          };
          // console.log("Track 3 =====", chatHist);

          // dispatch(updateUserChatDetails(chatHistObj));
          // scrollToBottom();
        }
      } catch (error) {
        setError("An error occurred. Please try again."); // Set error message
      } finally {
        const responsePostTrack: any = await client.post(ENDPOINTS.postTrack, {
          context_id: activeContextId.toString(),
          user_id: user_id.toString(),
          question: ques,
          question_id: qId,
          answer_id: aId,
          chat_id: activeChatId.toString(),
          content: postTrack1Resp,
          data_name: selectedDomain,
          track_1: postTrack1Resp,
          track_2: postTrack2Resp,
          track_3: postTrack3Resp,
        });
        const resPostTrackData: any = responsePostTrack.data;
        // console.log("response", resPostTrackData);
      }

      dispatch(addUserChatDetails(chatHist));
      scrollToBottom();

      // const payload: any = { user_id };

      // const res = await getUserChatDetailsReponse(payload);
      // dispatch(setUserChatDetails(res.data));
    } catch (error) {
      setError("An error occurred. Please try again."); // Set error message
    }

    dispatch(setIsFetchingQuesResponse(false)); // Set loading state to false
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (
      chatQuestion.trim() !== "" &&
      e.key === "Enter" &&
      !e.shiftKey &&
      !e.ctrlKey &&
      !e.altKey &&
      !e.metaKey &&
      !e.getModifierState("CapsLock")
    ) {
      e.preventDefault();
      sendQuestion();
    }
  };

  const createNewTopic = async () => {
    const newCtxtResp: newContextAPIResponse = await client.post(
      ENDPOINTS.newTopic,
      { user_id, chat_id: activeChatId, domain: selectedDomain }
    );
    const payload: any = { user_id, data_name: selectedDomain };
    const response = await getUserChatDetailsReponse(payload);
    dispatch(setUserChatDetails(response.data));
    dispatch(addNewContextId(newCtxtResp.data.context_id));
    dispatch(setOutletLoading(false));
  };

  return (
    <div className="w-[100%] bg-white">
      <div className="flex items-center w-[100%]">
        <button
          type="button"
          className="bg-[#18749C] flex items-center justify-center align-center w-[140px] h-[100%] rounded-[5px] py-[10px] text-[#FFFFFF] disabled:opacity-[0.5]"
          onClick={createNewTopic}
          disabled={isNewUser || isOldReport}
        >
          <AddIcon sx={{ color: "#FFFFFF", fontSize: 24 }} />
          <span className="ml-0.5 font-poppins font-regular text-[16px] text-[#FFFFFF]">
            New Topic
          </span>
        </button>
        <div className="flex w-[90.8%] h-[100%] ml-2 border border-solid border-[#EDEDED] rounded-[5px] shadow-[0_3px_6px_rgba(245,245,245,1)]">
          <Tooltip
            title={
              isOldReport
                ? "Asking follow up questions in already generated reports is not allowed at this moment."
                : ""
            }
          >
            <div className="flex align-center w-[100%] opacity-100">
              <TextareaAutosize
                className="focus:ring-0 border-none flex-grow resize-none scrollbar-thin scrollbar-thumb-gray-300 scrollbar-track-white"
                placeholder="Ask me anything..."
                value={chatQuestion}
                onChange={handleChange}
                onKeyDown={handleKeyDown}
                minRows={2}
                maxRows={2}
                style={{
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "flex-start",
                }}
                disabled={isOldReport}
              />
              <button
                type="button"
                className="p-[0.6%]"
                disabled={chatQuestion.trim() === "" || isOldReport}
                onClick={sendQuestion}
              >
                {/* {isLoading ? (
                <div className="flex items-center justify-center">
                  <div className="w-[10px] h-[10px] bg-[#18749C] rounded-full mr-1 animate-bounce" />
                  <div className="w-[10px] h-[10px] bg-[#18749C] rounded-full mr-1 animate-bounce" />
                  <div className="w-[10px] h-[10px] bg-[#18749C] rounded-full animate-bounce" />
                </div>
              ) : ( */}
                <img
                  className="w-[25px] h-[25px]"
                  src={chatQuestion.trim() === "" ? InvalidSend : ValidSend}
                  alt="send"
                />
                {/* )} */}
              </button>
            </div>
          </Tooltip>
        </div>
      </div>
    </div>
  );
}

export default ChatInput;
