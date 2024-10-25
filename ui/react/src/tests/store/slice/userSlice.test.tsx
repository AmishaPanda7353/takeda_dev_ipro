import { createSlice } from "@reduxjs/toolkit";
import { configureStore } from '@reduxjs/toolkit';
import userReducer, {
    setUser,
    setOutletLoading,
    setInitialUserID,
    setInitialUserChatState,
    setActiveChatIndex,
    setActiveContextId,
    setActiveReportChatId,
    setIsNewUser,
    setIsOldReport,
    setUserChatDetails,
    addUserChatDetails,
    setActiveChatDetails,
} from '../../../store/slice/userSlice';
describe('userSlice', () => {
    let store: any;

    beforeEach(() => {
        store = configureStore({
            reducer: {
                user: userReducer,
            },
        });
    });

    test('should set user', () => {
        const loggedIn = true;

        store.dispatch(setUser(loggedIn));

        const { loggedIn: result } = store.getState().user;
        expect(result).toEqual(loggedIn);
    });

    test('should set outlet loading', () => {
        const outletLoading = true;

        store.dispatch(setOutletLoading(outletLoading));

        const { outletLoading: result } = store.getState().user;
        expect(result).toEqual(outletLoading);
    });

    test('should set initial user ID', () => {
        const userId = '123';

        store.dispatch(setInitialUserID(userId));

        const { user_id: result } = store.getState().user;
        expect(result).toEqual(userId);
    });

    test('should set initial user chat state', () => {
        const chatDetails = [{
            "chat_id": 2,
            "user_id": "2",
            context_id: 2,
            "chat_name": "chat 9HF9VO",
            "data_name": 'Finance',
            "is_active": true,
            "chat_history":
            {
                "question_id": 1235,
                context_id: 1,
                "question": "Which shipment are delayed0?",
                "answer_id": 102,
                "type": "clarification",
                "data": {
                    "content": "I did not understand can you give me more business context"
                },
                "created_time": "2023-06-01 12:48:27"
            }
        }
        ];

        store.dispatch(setInitialUserChatState(chatDetails));

        const { chat_details: result } = store.getState().user;
        expect(result).toEqual(chatDetails);
    });

    test('should set active chat index', () => {
        const activeChatIndex = 2;

        store.dispatch(setActiveChatIndex(activeChatIndex));

        const { activeChatIndex: result } = store.getState().user;
        expect(result).toEqual(activeChatIndex);
    });

    test('should set active context ID', () => {
        const activeContextId = 3;

        store.dispatch(setActiveContextId(activeContextId));

        const { activeContextId: result } = store.getState().user;
        expect(result).toEqual(activeContextId);
    });

    test('should set active report chat ID', () => {
        const activeReportChatId = 4;

        store.dispatch(setActiveReportChatId(activeReportChatId));

        const { activeReportChatId: result } = store.getState().user;
        expect(result).toEqual(activeReportChatId);
    });

    test('should set isNewUser', () => {
        const isNewUser = true;

        store.dispatch(setIsNewUser(isNewUser));

        const { isNewUser: result } = store.getState().user;
        expect(result).toEqual(isNewUser);
    });

    test('should set isOldReport', () => {
        const isOldReport = true;

        store.dispatch(setIsOldReport(isOldReport));

        const { isOldReport: result } = store.getState().user;
        expect(result).toEqual(isOldReport);
    });

    // it('should add user chat details', () => {
    //     const chatHistoryObject = {
    //         "question_id": 1235,
    //         context_id: 1,
    //         "question": "Which shipment are delayed0?",
    //         "answer_id": 102,
    //         "type": "clarification",
    //         "data": {
    //             "content": "I did not understand can you give me more business context"
    //         },
    //         "created_time": "2023-06-01 12:48:27"
    //     };

    //     store.dispatch(addUserChatDetails(chatHistoryObject));

    //     const { chat_details, isNewUser, activeChatDetails } = store.getState().user;
    //     expect(chat_details[0].chat_history[0][0]).toEqual(chatHistoryObject);
    //     expect(isNewUser).toEqual(false);
    //     expect(activeChatDetails).toEqual([chat_details[0]]);
    // });

    // ... write more test cases for other actions


});
