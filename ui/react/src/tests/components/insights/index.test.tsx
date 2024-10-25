import React from 'react';
import { screen, fireEvent } from '@testing-library/react';
import Chatbot from 'pages/insights';
import renderWithProviders from 'tests/utils';
import '@testing-library/jest-dom/extend-expect';



jest.mock("react-plotly.js", () => ({
    __esModule: true,
    default: jest.fn(),
}));

describe('Chatbot', () => {

    test('should render the conversation component', () => {
        const { getByTestId } = renderWithProviders(<Chatbot />);
        const conversationComponent = getByTestId('conversation-component');
        expect(conversationComponent).toBeInTheDocument();
    });

});
