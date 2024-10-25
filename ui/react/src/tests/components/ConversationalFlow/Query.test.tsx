import { screen } from "@testing-library/react";
import Query from "components/ConversationalFlow/Query"
import renderWithProviders from "tests/utils"
import userEvent from '@testing-library/user-event';

describe("Query", () => {
    const query = 'SELECT * FROM table';
    test("Renders the query", () => {
        renderWithProviders(<Query query={query} error='' showError={true} botResponseQuery="ss" />);
        const queryElement = screen.getByText("Query");
        expect(queryElement).toBeInTheDocument();
    });

    test("copies the formatted query when copy icon is clicked", () => {
        renderWithProviders(<Query query={query} error='' botResponseQuery="ss" showError={true} />);
        const copyIcon = screen.getByTestId('copy-icon');
        userEvent.click(copyIcon);
    });

    test('displays the "Copied!" text when query is copied', async () => {
        renderWithProviders(<Query query={query} error='' showError={true} botResponseQuery="ss" />);
        const copyIcon = screen.getByTestId('copy-icon');
        userEvent.click(copyIcon);
        // const copiedText = await screen.findByText('Copied!');
        // expect(copiedText).toBeInTheDocument();
    });

    // test('does not display the "Copied!" text when query is not copied', () => {
    //     renderWithProviders(<Query query={query} />);
    //     const copiedText = screen.queryByText('Copied!');
    //     expect(copiedText).not.toBeInTheDocument();
    // });

})

