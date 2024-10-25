import React, { useState } from "react";
import { Column } from "@material-table/core";
import { Modal } from "@mui/material";
import TableButtons from "./TableButtons";
import TableListSelection from "./TableListSelection";
import NlpTable from "./NlpTable";
import { ChatHistoryObject } from "../ConversationalFlow/types"

type tableProps = {
    setModalShow: (flag: boolean) => void;
    modalShow: boolean;
    data: any;
    chatObject: ChatHistoryObject
}


function ChatTable({ setModalShow, modalShow, data }: tableProps) {


    return (

        <div >
            <Modal open={modalShow} onClose={() => setModalShow(false)}>
                <div className="bg-white absolute  rounded-[10px] top-1/2 left-1/2 transform -translate-x-2/4 -translate-y-2/4 p-[20px] w-[88%]  outline-none">
                    <div className="text-[#1C1C1C] font-semibold text-sm Poppins pb-[20px]">
                        Please select the most relevant tables for answering your question
                    </div>
                    <div className="flex pt-[40px] pl-[20px] pb-[40px] w-full justify-between border-solid border-[1px] rounded-[10px] ">
                        <div className="w-[75%] pr-[20px] border-[#DEE3EA] border-r-[3px] border-dashed">
                            {/* <NlpTable data={data} /> */}
                        </div>
                        <div className="w-[20%]">
                            <TableListSelection />
                        </div>
                    </div>
                    <TableButtons setModalShow={setModalShow} text="Continue" />
                </div>

            </Modal >

        </div>

    )
};


export default ChatTable