/* eslint-disable camelcase */
/* eslint-disable security/detect-object-injection */
import { createSlice } from "@reduxjs/toolkit";
import _ from "lodash";
import type { PayloadAction } from "@reduxjs/toolkit";
import {
  chatdetails,
  ReportObject,
  chatResponse,
  ChatHistoryObject,
  feedbackSetObject,
  mergeTrackDataObject,
} from "../../components/ConversationalFlow/types";

type InitialState = {
  loggedIn: boolean;
  outletLoading: boolean;
  user_id: string;
  chat_details: chatdetails[];
  activeChatDetails: chatdetails[];
  report_list: ReportObject[];
  activeChatIndex: number;
  activeContextId: number;
  activeChatId: number;
  isNewUser: boolean;
  activeReportChatId: number;
  isOldReport: boolean;
  isScrollBottom: boolean;
  isShowEmptySlice: boolean;
  isFetchingQuesResponse: boolean;
  selectedDomain: string;
  isSliderClosedICon: boolean;
  combineTrackData: mergeTrackDataObject[];
  false_count: number;
};

const storedAuthData = JSON.parse(localStorage.getItem("auth_state") as string);

const initialState: InitialState = {
  loggedIn: false,
  outletLoading: false,
  user_id: storedAuthData ? storedAuthData.userInfo.email : "",
  chat_details: [],
  activeChatDetails: [],
  report_list: [],
  activeChatIndex: 0,
  activeContextId: 0,
  activeChatId: 0,
  activeReportChatId: 0,
  isNewUser: true,
  isOldReport: false,
  isScrollBottom: true,
  isShowEmptySlice: false,
  isFetchingQuesResponse: false,
  selectedDomain: "",
  isSliderClosedICon: true,
  combineTrackData: [],
  false_count: 0,
};

export const userSlice = createSlice({
  name: "user",
  initialState,
  reducers: {
    setUser(state, action: PayloadAction<boolean>) {
      state.loggedIn = action.payload;
    },
    setOutletLoading(state, action: PayloadAction<boolean>) {
      state.outletLoading = action.payload;
    },
    setInitialUserID(state, action: PayloadAction<string>) {
      state.user_id = action.payload;
    },
    setInitialUserChatState(state, action: PayloadAction<chatdetails[]>) {
      state.chat_details = action.payload;
    },
    setActiveChatIndex(state, action: PayloadAction<number>) {
      state.activeChatIndex = action.payload;
    },
    setActiveContextId(state, action: PayloadAction<number>) {
      state.activeContextId = action.payload;
    },
    setActiveReportChatId(state, action: PayloadAction<number>) {
      state.activeReportChatId = action.payload;
    },
    setIsNewUser(state, action: PayloadAction<boolean>) {
      state.isNewUser = action.payload;
    },
    setIsOldReport(state, action: PayloadAction<boolean>) {
      state.isOldReport = action.payload;
    },
    setIsEmptySlice(state, action: PayloadAction<boolean>) {
      state.isShowEmptySlice = action.payload;
    },
    setIsFetchingQuesResponse(state, action: PayloadAction<boolean>) {
      state.isFetchingQuesResponse = action.payload;
    },
    setUserChatDetails(state, action: PayloadAction<chatResponse>) {
      // let activeIx: number = _.findIndex(action.payload.chat_details, (i) => i.is_active === true) || 0;
      // activeIx = activeIx === -1 ? 0 : activeIx
      const activeContext = action.payload.chat_details.context_id; // [activeIx]
      state.chat_details = [action.payload.chat_details];
      state.activeChatIndex = 0;
      state.activeChatDetails = [action.payload.chat_details]; // [activeIx]
      state.activeChatId = action.payload.chat_details.chat_id; // [activeIx]
      state.activeReportChatId = action.payload.chat_details.chat_id; // [activeIx]
      state.activeContextId = activeContext;
      state.isNewUser = !(_.size(action.payload.chat_details.chat_history) > 0); // [activeIx]
      state.report_list = [
        {
          report_id: 0,
          report_name: "Active",
          report_url: "",
          chat_id: action.payload.chat_details.chat_id, // [activeIx]
        },
        ...action.payload.report_list,
      ];
      state.isScrollBottom = true;
    },
    addNewContextId(state, action: PayloadAction<number>) {
      const chatDetails = [...state.chat_details];
      const activeChatDetails = [...state.activeChatDetails];
      if (!chatDetails[state.activeChatIndex].chat_history[action.payload]) {
        chatDetails[state.activeChatIndex].chat_history[action.payload] = [];
      }
      if (
        !activeChatDetails[state.activeChatIndex].chat_history[action.payload]
      ) {
        activeChatDetails[state.activeChatIndex].chat_history[action.payload] =
          [];
      }
      state.chat_details = chatDetails;
      state.activeChatDetails = activeChatDetails;
      state.isShowEmptySlice = true;
    },
    updateUserChatDetails(state, action: PayloadAction<ChatHistoryObject>) {
      const chatDetails = [...state.chat_details];
      const { context_id, question_id } = action.payload;
      const dataArr: any = [...state.combineTrackData];

      // Find the chat history for the current context
      let chatHistory = [
        ...chatDetails[state.activeChatIndex].chat_history[context_id],
      ];

      // Check if this is a new question or an answer
      if (question_id !== 0) {
        // Add the question to the chat history
        const tempChatHistory: any = [];
        tempChatHistory.push(action.payload);
        chatHistory = tempChatHistory.map((chat: any) => {
          if (chat.question_id === action.payload.question_id) {
            return {
              ...chat,
              data: { ...chat?.data, data: dataArr },
            };
          }
          return chat;
        });
      }

      // Update the chat history for the current context
      chatDetails[state.activeChatIndex].chat_history[context_id] = chatHistory;

      // Update state
      state.chat_details = chatDetails;
      state.isNewUser = !(chatHistory.length > 0);
      state.activeChatDetails = [chatDetails[state.activeChatIndex]];
      state.isScrollBottom = true;
    },

    addUserChatDetails(state, action: PayloadAction<ChatHistoryObject>) {
      const chatDetails = [...state.chat_details];
      let chatHistory = [...chatDetails[state.activeChatIndex].chat_history[state.activeContextId]];
      if (chatDetails[state.activeChatIndex].chat_history[state.activeContextId]) {
        chatHistory = _.filter(chatHistory, (i) => i.question_id !== 0 && i.answer_id !== 0);
        chatHistory.push(action.payload);
      } else {
        chatHistory = [];
        chatHistory = _.filter(chatHistory, (i) => i.question_id !== 0 && i.answer_id !== 0);
        chatHistory.push(action.payload);
      }
      chatDetails[state.activeChatIndex].chat_history[state.activeContextId] = chatHistory;
      state.chat_details = chatDetails;
      state.isNewUser = !(_.size(chatDetails[state.activeChatIndex].chat_history) > 0);
      state.activeChatDetails = [chatDetails[state.activeChatIndex]];
      state.isScrollBottom = true;
    },

    addUserChatQuestionDetails(
      state,
      action: PayloadAction<ChatHistoryObject>
    ) {
      const chatDetails = [...state.chat_details];
      if (
        chatDetails[state.activeChatIndex].chat_history[state.activeContextId]
      ) {
        chatDetails[state.activeChatIndex].chat_history[
          state.activeContextId
        ].push(action.payload);
      } else {
        chatDetails[state.activeChatIndex].chat_history[state.activeContextId] =
          [];
        chatDetails[state.activeChatIndex].chat_history[
          state.activeContextId
        ].push(action.payload);
      }
      state.chat_details = chatDetails;
      state.isNewUser = !(
        _.size(chatDetails[state.activeChatIndex].chat_history) > 0
      );
      state.activeChatDetails = [chatDetails[state.activeChatIndex]];
      state.isScrollBottom = true;
    },
    setActiveChatDetails(state, action: PayloadAction<chatdetails[]>) {
      state.activeChatDetails = action.payload;
      state.isScrollBottom = true;
    },
    setAnswerFeedback(state, action: PayloadAction<feedbackSetObject>) {
      const activeChatDets = [...state.activeChatDetails];
      const allChatDets = [...state.chat_details];
      let acIx = -1;
      let alIx = -1;
      if (activeChatDets[0].chat_history[action.payload.context_id]) {
        acIx = activeChatDets[0].chat_history[
          action.payload.context_id
        ].findIndex(
          (i) => i.answer_id.toString() === action.payload.answer_id.toString()
        );
      }
      if (allChatDets[0].chat_history[action.payload.context_id]) {
        alIx = allChatDets[0].chat_history[action.payload.context_id].findIndex(
          (i) => i.answer_id.toString() === action.payload.answer_id.toString()
        );
      }
      if (acIx !== -1) {
        activeChatDets[0].chat_history[action.payload.context_id][
          acIx
        ].feedback = action.payload.isLiked;
      }
      if (alIx !== -1) {
        allChatDets[0].chat_history[action.payload.context_id][acIx].feedback =
          action.payload.isLiked;
      }
      state.chat_details = allChatDets;
      state.activeChatDetails = activeChatDets;
      state.isScrollBottom = false;
    },
    setSelectedDomain(state, action: PayloadAction<string>) {
      state.selectedDomain = action.payload;
    },
    setIsSliderClosedICon(state, action: PayloadAction<boolean>) {
      state.isSliderClosedICon = action.payload;
    },

    setCombineTrackData(state, action: PayloadAction<mergeTrackDataObject[]>) {
      state.combineTrackData = action.payload;
    },

    setFalseCount(state, action: PayloadAction<number>) {
      state.false_count = action.payload;
    },
  },
});

export const {
  setUser,
  setInitialUserID,
  setInitialUserChatState,
  setActiveChatIndex,
  setActiveReportChatId,
  setIsOldReport,
  setActiveContextId,
  setIsNewUser,
  setUserChatDetails,
  updateUserChatDetails,
  addUserChatDetails,
  setOutletLoading,
  setActiveChatDetails,
  setAnswerFeedback,
  addUserChatQuestionDetails,
  addNewContextId,
  setIsFetchingQuesResponse,
  setSelectedDomain,
  setIsSliderClosedICon,
  setCombineTrackData,
  setFalseCount,
} = userSlice.actions;

export default userSlice.reducer;
