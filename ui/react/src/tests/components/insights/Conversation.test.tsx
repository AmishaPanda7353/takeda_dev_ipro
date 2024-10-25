import Conversation from "pages/insights/Conversation";
import renderWithProviders from "tests/utils";


jest.mock("react-plotly.js", () => ({
    __esModule: true,
    default: jest.fn(),
}));

const scrollEndRef = jest.fn();
const scrollToBottom = jest.fn();

describe("Conversation", () => {
    test('renders component without chat history', () => {
        renderWithProviders(<Conversation scrollEndRef={scrollEndRef} scrollToBottom={scrollToBottom} />);
        // Add assertions for the absence of chat history elements
    });
})