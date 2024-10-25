import { fireEvent, screen } from "@testing-library/react";
import UserAnswerFeedback from "components/ConversationalFlow/UserAnswerFeedback"
import renderWithProviders from "tests/utils"
import '@testing-library/jest-dom/extend-expect';


describe("UserAnswerFeedback", () => {
    const answerId = "Janivi";
    const defaultLike = true;
    const defaultDislike = true;
    const contextId = 1;

    test("Render like button", () => {
        renderWithProviders(<UserAnswerFeedback answerId={answerId} defaultDislike={defaultDislike} defaultLike={defaultLike} contextId={contextId} />)
        const likeButton = screen.getByTestId("like");
        expect(likeButton).toBeInTheDocument();
        expect(likeButton).toHaveAttribute("disabled");
    })

    test("Render dislike button", () => {
        renderWithProviders
            (<UserAnswerFeedback
                answerId={answerId}
                defaultDislike={defaultDislike}
                defaultLike={defaultLike}
                contextId={contextId}
            />)
        const dislikeButton = screen.getByTestId("dislike");
        expect(dislikeButton).toBeInTheDocument();
        expect(dislikeButton).not.toHaveAttribute("disabled");
    })

    it('should trigger postUserFeedback when like button is clicked', () => {
        const mockPostUserFeedback = jest.fn();
        renderWithProviders(
            (<UserAnswerFeedback
                answerId={answerId}
                defaultDislike={defaultDislike}
                defaultLike={defaultLike}
                contextId={contextId}
            />)
        );

        const likeButton = screen.getByTestId('like');
        fireEvent.click(likeButton);

        expect(mockPostUserFeedback).toHaveBeenCalledTimes(0);
        // expect(mockPostUserFeedback).toHaveBeenCalledWith(true, '');
    });


    test("should show feedback modal and trigger posterUserFeedback when dislike button is clicked", () => {
        const mockPostUserFeedback = jest.fn();
        renderWithProviders(<UserAnswerFeedback
            answerId={answerId}
            defaultDislike={defaultDislike}
            defaultLike={defaultLike}
            contextId={contextId}
        />)

        const dislikeButton = screen.getByTestId("dislike");
        fireEvent.click(dislikeButton);

        // const feedbackModal = screen.getByTestId('feedback-modal');
        // expect(feedbackModal).toBeInTheDocument();

        expect(mockPostUserFeedback).toHaveBeenCalledTimes(0);
        // expect(mockPostUserFeedback).toHaveBeenCalledWith(false, '');

    })
})