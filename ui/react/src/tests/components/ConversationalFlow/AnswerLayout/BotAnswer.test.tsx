import { fireEvent, screen } from "@testing-library/react";
import React, { Children } from "react";
import AnswerLayout from "components/ConversationalFlow/AnswerLayout/AnswerLayout";
import Feedback from "components/FeedbackModal";
import renderWithProviders from "../../../utils";
import BotAnswer from "../../../../components/ConversationalFlow/AnswerLayout/BotAnswer"


const likeStatus = true;
const text = "Hello"
const like = true;
const dislike = true;
const setLike = jest.fn()
const setDisLike = jest.fn()
const setFeedbackText = jest.fn()

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

test("render the BotAnswer", () => {
    renderWithProviders(<BotAnswer text={text} chatObject={chatObject} />);

    // const likeId = likeStatus && screen.getByTestId('like');
    expect(screen.getByRole("img", { name: "chatbotICon" })).toBeInTheDocument
    const likebtn = screen.getByTestId("like")
    expect(fireEvent.click(likebtn))
    // expect(setLike).toBeCalled();
    const disLikebtn = screen.getByTestId("dislike")
    expect(fireEvent.click(disLikebtn))
    // expect(setDisLike).toBeCalled();

})


test("render the AnswerLayout", () => {
    renderWithProviders(<AnswerLayout> </AnswerLayout>)
})
test("render the feedback", () => {
    dislike && renderWithProviders(<Feedback feedbackModalShow={dislike} setfeedbackModalShow={setDisLike} setFeedbackText={setFeedbackText} />)
})

// test('toggle like state on click', () => {
//     // Render the component and set the initial state
//     const setLike = jest.fn(); // Create a mock function
//     const like = false; // Set initial state to false
//     // renderWithProviders(<MyComponent like={like} setLike={setLike} />);

//     // Assert initial state
//     expect(setLike).not.toHaveBeenCalled(); // Ensure that the mock function has not been called

//     // Simulate button click
//     const likeButton = screen.getByTestId('like');
//     fireEvent.click(likeButton);

//     // Assert updated state
//     expect(setLike).toHaveBeenCalledWith(true); // Ensure that the mock function has been called with the expected value
// });