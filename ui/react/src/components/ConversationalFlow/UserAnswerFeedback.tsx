/* eslint-disable camelcase */
import React, { useState } from 'react'
import Feedback from 'components/FeedbackModal';
import client from 'client';
import { useAppDispatch, useAppSelector } from "store";
import { setOutletLoading, setAnswerFeedback } from 'store/slice/userSlice';
import { ENDPOINTS } from 'constants/constant';
// import { getUserChatDetailsReponse } from 'constants/common';
import likeImage from "../../assets/images/LikeImage.svg"
import GreenlikeImage from "../../assets/images/GreenLikeImage.svg"
import DislikeImage from "../../assets/images/DislikeImage.svg"
import RedDislikeImage from "../../assets/images/RedDislikeImage.svg"


function UserAnswerFeedback({ answerId, defaultLike, defaultDislike, contextId }:
  { answerId: string, defaultLike: boolean, defaultDislike: boolean, contextId: number }) {
  const dispatch = useAppDispatch();
  const { selectedDomain } = useAppSelector((s) => s.user);
  const [like, setLike] = useState<boolean>(defaultLike);
  const [disLike, setDisLike] = useState<boolean>(defaultDislike);
  const [showFeedbackModal, setShowFeedbackModal] = useState<boolean>(false);

  const postUserFeedback = async (isLikeClicked: boolean, feedbackText: string) => {
    const res = await client.post(ENDPOINTS.chatFeedback,
      {
        status: isLikeClicked,
        answer_id: answerId,
        reason: feedbackText,
        data_name: selectedDomain
      });
    if (res.data?.status) {
      dispatch(setAnswerFeedback({ answer_id: parseInt(answerId, 10), isLiked: isLikeClicked, context_id: contextId }));
    }
    // const payload: any = { user_id };
    // const response = await getUserChatDetailsReponse(payload);
    // dispatch(setUserChatDetails(response.data));
    dispatch(setOutletLoading(false));
  }

  const handleClickUserFeedback = async (isLikeClicked: boolean) => {
    if (isLikeClicked) {
      setLike(like)
      postUserFeedback(isLikeClicked, '');
    } else {
      setShowFeedbackModal(true)
      setDisLike(disLike)
    }
  }

  return (
    <>
      <button type="button" disabled={defaultDislike} data-testid="like" className='pr-[10px] disabled:opacity-50' onClick={() => handleClickUserFeedback(true)}>
        {like ? <img src={GreenlikeImage} alt="Like" /> : <img src={likeImage} alt="Like" />}
      </button>
      <button type="button" data-testid="dislike" className='disabled:opacity-50' onClick={() => handleClickUserFeedback(false)}>
        {disLike ? <img className='disabled:opacity-50' src={RedDislikeImage} alt="Dislike" /> : <img src={DislikeImage} alt="Dislike" />}
      </button>
      {
        showFeedbackModal && !defaultDislike &&
        <Feedback feedbackModalShow={showFeedbackModal}
          setfeedbackModalShow={setShowFeedbackModal}
          setFeedbackText={postUserFeedback}
        />
      }
    </>
  )
}

export default UserAnswerFeedback;