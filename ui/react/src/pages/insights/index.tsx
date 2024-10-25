/* eslint-disable react/no-array-index-key */
/* eslint-disable security/detect-object-injection */
/* eslint-disable camelcase */
import React, { useRef, useState } from 'react';
import { useAppDispatch, useAppSelector } from 'hooks/store-hooks'
import { setIsFetchingQuesResponse } from 'store/slice/userSlice';
import GenerateReport from './GenerateReport';
import Conversation from './Conversation';
import ChatInput from './ChatInput';
import BouncingDotsLoader from './BouncingDotsLoader';



function Chatbot() {
  const scrollEndRef = useRef<HTMLDivElement>(null);
  // const [isLoading, setIsLoading] = useState(false); // State to track loading status
  const dispatch = useAppDispatch();
  const { isFetchingQuesResponse, isOldReport } = useAppSelector((s) => s.user);

  const setIsLoading = (flag: boolean) => {
    dispatch(setIsFetchingQuesResponse(flag))
  }

  const scrollToBottom = () => {
    if (scrollEndRef.current) {
      const scrollContainer = scrollEndRef.current;
      scrollContainer.scrollTop = scrollContainer.scrollHeight;
    }
  };

  return (
    <div data-testid="conversation-component" className="w-[100%] bg-white" >
      <Conversation scrollEndRef={scrollEndRef} scrollToBottom={scrollToBottom} />
      {isFetchingQuesResponse && !isOldReport && <BouncingDotsLoader />}
      <GenerateReport />
      <ChatInput scrollToBottom={scrollToBottom} isLoading={isFetchingQuesResponse} />
    </div>
  )
}

export default Chatbot