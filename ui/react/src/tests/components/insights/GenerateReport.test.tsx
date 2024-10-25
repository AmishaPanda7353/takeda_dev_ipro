import { cleanup, fireEvent, screen } from "@testing-library/react";
import GenerateReport from "pages/insights/GenerateReport";
import renderWithProviders from "tests/utils";
import '@testing-library/jest-dom/extend-expect';

afterEach(cleanup);

jest.mock("@mui/material", () => ({
    ...jest.requireActual("@mui/material"),
    Modal: ({ children, open, onClose }: { children: any, open: any, onClose: any }) =>
        open ? <div>{children}</div> : null,
}));

describe("GenerateReport", () => {
    test("renders GenerateReport component", () => {
        renderWithProviders(<GenerateReport />);

        const generateReportButton = screen.getByText("Generate Report");
        expect(generateReportButton).toBeInTheDocument();
        expect(generateReportButton).toBeEnabled();
    });

    test("opens the modal when Generate Report button is clicked", () => {
        renderWithProviders(<GenerateReport />);

        const generateReportButton = screen.getByText("Generate Report");
        fireEvent.click(generateReportButton);

    });

    test("does not render the modal initially", () => {
        renderWithProviders(<GenerateReport />);

        const modalTitle = screen.queryByText("Reports");
        expect(modalTitle).toBeNull();
    });
});
