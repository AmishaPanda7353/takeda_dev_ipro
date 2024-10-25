import { cleanup, fireEvent, render, screen } from "@testing-library/react"
import Feedback from "components/FeedbackModal"
import renderWithProviders from "tests/utils"
import { Formik, Form, Field } from 'formik';

afterEach(cleanup)
const feedbackModalShow = true
const setfeedbackModalShow = jest.fn();
const setFeedbackText = jest.fn();

test("render feedback modal", () => {

    console.log = jest.fn();
    window.alert = jest.fn();

    renderWithProviders(<Feedback feedbackModalShow={feedbackModalShow} setfeedbackModalShow={setfeedbackModalShow} setFeedbackText={setFeedbackText} />);

    const closeBtn = screen.getByRole("button", { name: 'Cancel' });
    fireEvent.click(closeBtn);
    expect(setfeedbackModalShow).toBeCalled();

    expect(screen.getByRole('img', { name: 'dislike' })).toBeInTheDocument
    // expect(screen.getByLabelText("Tell us More")).toBeInTheDocument
    const submitBtn = screen.getByRole("button", { name: 'Submit' });
    fireEvent.click(submitBtn);


})



test('form submission and alert message', () => {
    const initialValues = {
        issue: [],
        description: "",
    }
    // const setSubmitting = jest.fn();

    renderWithProviders(<Formik initialValues={initialValues} onSubmit={(values, actions) => { actions.setSubmitting(false) }} />);


    // Submit the form
    // const submitButton = screen.getByText(/submit/i);
    // fireEvent.click(submitButton);

});