/* eslint-disable react/no-array-index-key */
/* eslint-disable security/detect-object-injection */
/* eslint-disable camelcase */
import React, { useEffect } from 'react';
import _ from 'lodash'
import { useAppSelector } from "store";
import BotAnswer from 'components/ConversationalFlow/AnswerLayout/BotAnswer';
import Questions from 'components/ConversationalFlow/QuestionLayout/Questions';
import InsightType from 'components/ConversationalFlow/InsightTypes';
import { chatdetails } from 'components/ConversationalFlow/types';


function Conversation({ scrollEndRef, scrollToBottom }: { scrollEndRef: any, scrollToBottom: any }) {
  const { activeChatDetails, isScrollBottom, isShowEmptySlice } = useAppSelector((s) => s.user);

  const activeChat: chatdetails = activeChatDetails[0];

  useEffect(() => {
    if (isScrollBottom) {
      scrollToBottom();
    }
  }, [activeChat]);

  return (
    <div className='bg-white h-[68vh] scrollbar-thin scrollbar-thumb-gray-400 scrollbar-track-white' ref={scrollEndRef}>
      {
        activeChat?.chat_history && _.size(activeChat?.chat_history) > 0 ? (
          <>
            {
              _.map(activeChat?.chat_history, (history, key) => (
                <div className="w-[100%] bg-[#F4F8FD] rounded-[10px] pb-[10px] pt-[10px] mb-[15px]" key={key}>
                  <div className='' >
                    {history.map((val, idx) => {
                      let contentData: any = val.data;
                      contentData = _.has(val.data, 'data') ? val?.data?.data : val.data;
                      return (
                        <div className='' key={`${val.question_id}-${idx}`}>
                          <Questions text={val.question} chatObject={val} />
                          {
                            val.type.toLowerCase() === "clarification" ?
                              <BotAnswer text={contentData.content} chatObject={val} /> :
                              val.type.toLowerCase() === "insight" || val.type.toLowerCase() === "insights" ?
                                <InsightType data={contentData} chatObject={val} />
                                :
                                // val.type.toLowerCase() === "table_selector" ?
                                //   <div className='border-[#DEE3EA] mt-[20px]  border-2 rounded-[5px] ml-[8%] mr-[6%]'>
                                //     <NlpTable data={contentData.content} />
                                //   </div> :
                                ""
                          }
                        </div>
                      )
                    })}
                  </div>
                </div>
              ))
            }
          </>

        ) : (
          <div className={`w-[100%] bg-[#F4F8FD] rounded-[10px] pb-[10px] pt-[10px] mb-[15px] ${isShowEmptySlice ? '' : 'hidden'}`}>
            <div className='h-[1vh] bg-white mx-3' />
          </div>
        )
      }

    </div>
  )
}

export default Conversation;