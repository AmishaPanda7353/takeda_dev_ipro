/* eslint-disable camelcase */
import { useAppSelector, useAppDispatch } from "store";
import { Modal } from '@mui/material'
import client from 'client';
import { setOutletLoading, setUserChatDetails } from 'store/slice/userSlice';
import { getUserChatDetailsReponse } from 'constants/common';
import { ENDPOINTS } from 'constants/constant';

type ResetChatProp = {
  resetModalShow: boolean;
  setResetModalShow: (flag: boolean) => void;
}

function ResetChat({ resetModalShow, setResetModalShow }: ResetChatProp) {
  const dispatch = useAppDispatch();
  const { user_id, activeChatId, isOldReport, selectedDomain } = useAppSelector((s) => s.user);
  const resetChat = async () => {
    dispatch(setOutletLoading(true));
    await client.post(ENDPOINTS.resetChat, { user_id, chat_id: activeChatId, domain: selectedDomain });
    const payload: any = { user_id, data_name: selectedDomain };
    const response = await getUserChatDetailsReponse(payload);
    dispatch(setUserChatDetails(response.data));
    dispatch(setOutletLoading(false));
    setResetModalShow(false);
  }

  return (
    <Modal open={resetModalShow}>
      <div data-testid="reset-chat-modal" className="bg-white absolute rounded-[10px] top-1/2 left-1/2 transform -translate-x-2/4 -translate-y-2/4 p-5 w-[536px] h-[210px] outline-none">
        <div className="flex font-poppins font-semibold text-[16px] text-black mb-[25px]">Reset Chat?</div>
        <div className="font-poppins  text-[16px] text-black">
          {/* <p>After starting new chat you cannot edit your previous conversation.</p>
          <p>But you can see your previous conversation in report.</p> */}

          <p>You will not be able to see the conversations again.This will clear all conversations in this report.</p>
        </div>
        <div className="flex justify-end gap-5 mt-[30px] font-medium">
          <button type="button"
            className="font-poppins rounded bg-white text-[#18749C] border-[#18749C] border-[1px] h-[40px] w-[90px]"
            onClick={() => { setResetModalShow(false) }}>
            Cancel
          </button>
          <button type="button" className={`${isOldReport ? "opacity-50" : ""} font-poppins rounded text-white bg-[#18749C] h-[40px] w-[90px]`}
            onClick={() => resetChat()}
            disabled={isOldReport}
          >
            Reset
          </button>
        </div>
      </div>
    </Modal>
  )
}

export default ResetChat;