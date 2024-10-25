import { cleanup, fireEvent, screen, waitFor } from "@testing-library/react";
import InsightType from "components/ConversationalFlow/InsightTypes"
import renderWithProviders from "tests/utils"
import MaterialTable, { Column } from "@material-table/core";

jest.mock("react-plotly.js", () => ({
    __esModule: true,
    default: jest.fn(),
}));

// Mock the MaterialTable component
jest.mock('@material-table/core', () => {
    return jest.fn().mockImplementation(({ data }: { data: any }) => {
        return <div>Mocked MaterialTable Component</div>;
    });
});

describe('InsightType', () => {

    const chatObject = {
        created_time: '2022-01-01T00:00:00Z',
        question_id: 2235,
        context_id: 3,
        feedback: false,
        question: "Where is table data434?",
        answer_id: 105,
        type: "table_selector",
        data: {
            "content": "I did not understand can you give me more business context"
        },
    }

    const data = [
        { insight_type: 'summary', content: 'Summary content' },
        { insight_type: 'sql_query', content: 'SELECT * FROM table' },
        { insight_type: 'chart', content: { data: [{ x: [1, 2, 3], y: [4, 5, 6], type: 'scatter' }], layout: { title: 'Chart' } } },
        { insight_type: 'table_selector', content: [{ id: 1, name: 'Rohan' }, { id: 2, name: 'Rohini' }] },
        { insight_type: 'number', content: 33 },
    ];



    test("Render Insight", () => {
        renderWithProviders(<InsightType data={data} chatObject={chatObject} />)
        // await waitFor(() => {

        //     expect(screen.getByText('Summary content')).toBeInTheDocument();
        //     expect(screen.getByText('SELECT * FROM table')).toBeInTheDocument();
        //     expect(screen.getByText('Chart')).toBeInTheDocument();
        //     expect(screen.getByText('Rohan')).toBeInTheDocument();
        //     expect(screen.getByText('Rohini')).toBeInTheDocument();
        //     expect(screen.getByText('33')).toBeInTheDocument();
        // })


    })
})

