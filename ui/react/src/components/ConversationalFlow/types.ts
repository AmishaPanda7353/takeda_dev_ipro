export type ReportObject = {
    report_id: number;
    report_name: string;
    report_url: string;
    chat_id: number;
  };
  
  export type IndiviualDataInsightType = {
    insight_type: string;
    content: string;
  };
  
  export type ClarificationInsightType = {
    content: string;
  };
  
  export type ChatHistoryObject = {
    question_id: number;
    context_id: number;
    feedback: boolean | null;
    question: string;
    answer_id: number;
    type: string;
    // data: IndiviualDataInsightType[] | ClarificationInsightType,
    data: any;
    created_time: string;
    error?: string;
    status?: string[];
    isWaiting?: boolean;
  };
  
  export type chatdetails = {
    chat_id: number;
    context_id: number;
    user_id: string;
    chat_name: string;
    data_name: string;
    is_active: boolean;
    chat_history: {
      [key: number]: ChatHistoryObject[];
    };
  };
  
  export type chatResponse = {
    user_id: string;
    report_list: ReportObject[];
    chat_details: chatdetails;
  };
  
  export type chatDetailsAPIResponse = {
    data: chatResponse;
  };
  
  export type questionAPIResponse = {
    data: any;
  };
  
  export type chatHistoryAPIResponse = {
    data: {
      [key: number]: ChatHistoryObject[];
    };
  };
  
  export type newContextAPIResponse = {
    data: {
      context_id: number;
    };
  };
  
  export type feedbackSetObject = {
    answer_id: number;
    isLiked: boolean;
    context_id: number;
  };
  
  export type mergeTrackDataObject = {
    insight_type: string;
    content: string;
    error: string;
    showError: boolean;
    table: any;
    bot_response: string;
  };
