import Questions from "components/ConversationalFlow/QuestionLayout/Questions"
import renderWithProviders from "tests/utils";

const chatObject = {
    "question_id": 1235,
    context_id: 1,
    "feedback": false,
    "question": "Which shipment are delayed0?",
    "answer_id": 102,
    "type": "clarification",
    "data": {
        "content": "I did not understand can you give me more business context"
    },
    "created_time": "2023-06-01 12:48:27"
}
test("render the Question component", () => {
    renderWithProviders(<Questions text="Hello" chatObject={chatObject} />)
})