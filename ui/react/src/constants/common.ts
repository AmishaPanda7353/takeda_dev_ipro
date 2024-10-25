/* eslint-disable import/prefer-default-export */
import client from "client";
import { chatDetailsAPIResponse } from "components/ConversationalFlow/types";
import { CHATRESPONSE, ENDPOINTS } from "./constant";

export const getUserChatDetailsReponse = async (payload: { payload: any }) => {
    const response: chatDetailsAPIResponse = await client.post(ENDPOINTS.chatDetails, payload); // { data: CHATRESPONSE }; 
    return response;
};
