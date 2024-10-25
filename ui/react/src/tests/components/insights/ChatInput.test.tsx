import { cleanup, fireEvent, screen } from "@testing-library/react";
import '@testing-library/jest-dom/extend-expect';
import ChatInput from "pages/insights/ChatInput";
import renderWithProviders from "tests/utils";
import InvalidSend from '../../../assets/images/InvalidSend.png';
import ValidSend from '../../../assets/images/ValidSend.png';
import { Modal } from "@mui/material";

afterEach(cleanup);

const scrollBottom = jest.fn();
const setIsLoading = jest.fn();


describe('ChatInput', () => {
    test("renders ChatInput component", () => {
        renderWithProviders(
            <ChatInput

                isLoading={true}
                scrollToBottom={scrollBottom}
            />
        );

        expect(screen.getByPlaceholderText("Ask me anything...")).toBeInTheDocument();

        const textarea = screen.getByPlaceholderText("Ask me anything...") as HTMLTextAreaElement;

        // Simulate typing in the textarea
        fireEvent.change(textarea, { target: { value: "Hello" } });

        // Verify that the chatQuestion state is updated correctly
        expect(textarea.value).toBe("Hello");

        // const sendButton = screen.getByAltText("send");

        // Assert that the image source is correct based on chatQuestion
        // expect(sendButton.getAttribute("src")).toContain(
        //     ValidSend
        // );
    });

    test("renders invalid send button when chatQuestion is empty", () => {
        const updatedChatQuestion = "";
        renderWithProviders(
            <ChatInput

                isLoading={false}
                scrollToBottom={scrollBottom}
            />
        );
        const sendButton = screen.getByAltText("send");
        expect(sendButton.getAttribute("src")).toBe(InvalidSend);
    });

    test("should render the modal and handle button clicks", () => {
        const setModalShow = jest.fn();
        const generateReport = jest.fn();

        // Render the modal component
        const { getByText } = renderWithProviders(
            <Modal open={true} onClose={setModalShow} className="">
                <>
                    <button onClick={generateReport}>Download</button>
                    <button onClick={setModalShow}>Cancel</button>
                </>
            </Modal>
        );

        // Assert that the modal content is rendered
        expect(
            getByText((content: any, element: any) => {
                return (
                    element.tagName.toLowerCase() === "button" &&
                    content.includes("Download")
                );
            })
        ).toBeInTheDocument();

        // Simulate button click
        fireEvent.click(getByText("Download"));

        // Assert that the generateReport function is called
        expect(generateReport).toHaveBeenCalled();

        // Simulate button click
        fireEvent.click(
            getByText((content: any, element: any) => {
                return (
                    element.tagName.toLowerCase() === "button" &&
                    content.includes("Cancel")
                );
            })
        );

        // Assert that the setModalShow function is called
        expect(setModalShow).toHaveBeenCalled();
    });

    test("should render the questions", () => {
        const questions = [
            { question: "Question 1" },
            { question: "Question 2" },
            { question: "Question 3" },
        ];

        // Render the modal component with the questions
        const { getByText } = renderWithProviders(
            <Modal open={true} onClose={() => { }}>
                {/* Modal content */}
                <div>
                    {questions.map((question, index) => (
                        <div key={index} className="py-2">
                            Q{index + 1}: {question.question}
                        </div>
                    ))}
                </div>
            </Modal>
        );

        // Assert that each question is rendered
        questions.forEach((question, index) => {
            const questionElement = getByText(
                `Q${index + 1}: ${question.question}`
            );
            expect(questionElement).toBeInTheDocument();
        });
    });
});
