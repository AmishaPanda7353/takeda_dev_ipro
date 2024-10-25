import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import NlpTable from "components/table/NlpTable";
import renderWithProviders from "tests/utils";

afterEach(cleanup);

const ChatHistoryObject = {
    question_id: 1,
    context_id: 2,
    feedback: true,
    question: "sample",
    answer_id: 1,
    type: "Hello",
    // data: IndiviualDataInsightType[] | ClarificationInsightType,
    data: "",
    created_time: "23:06:2023",
}

const data = [
    [
        { firstCol: "Data", secondCol: "Data", thirdCol: 50, fourthCol: "Data", fifthCol: 50, sixthCol: "Data", seventhCol: "Data" },
        { firstCol: "Data", secondCol: "Data", thirdCol: 50, fourthCol: "Data", fifthCol: 50, sixthCol: "Data", seventhCol: "Data" },
        { firstCol: "Data", secondCol: "Data", thirdCol: 50, fourthCol: "Data", fifthCol: 50, sixthCol: "Data", seventhCol: "Data" },
        { firstCol: "Data", secondCol: "Data", thirdCol: 50, fourthCol: "Data", fifthCol: 50, sixthCol: "Data", seventhCol: "Data" },
        { firstCol: "Data", secondCol: "Data", thirdCol: 50, fourthCol: "Data", fifthCol: 50, sixthCol: "Data", seventhCol: "Data" },
    ],
    [
        { firstCol: "Data", secondCol: "Data", thirdCol: 10, fourthCol: "Data", fifthCol: 50, sixthCol: "Data", seventhCol: "Data" },
        { firstCol: "Data", secondCol: "Data", thirdCol: 10, fourthCol: "Data", fifthCol: 50, sixthCol: "Data", seventhCol: "Data" },
        { firstCol: "Data", secondCol: "Data", thirdCol: 10, fourthCol: "Data", fifthCol: 50, sixthCol: "Data", seventhCol: "Data" },
        { firstCol: "Data", secondCol: "Data", thirdCol: 10, fourthCol: "Data", fifthCol: 50, sixthCol: "Data", seventhCol: "Data" },
        { firstCol: "Data", secondCol: "Data", thirdCol: 50, fourthCol: "Data", fifthCol: 50, sixthCol: "Data", seventhCol: "Data" },
    ],
    [
        { firstCol: "Data", secondCol: "Data", thirdCol: 20, fourthCol: "Data", fifthCol: 50, sixthCol: "Data", seventhCol: "Data" },
        { firstCol: "Data", secondCol: "Data", thirdCol: 20, fourthCol: "Data", fifthCol: 50, sixthCol: "Data", seventhCol: "Data" },
        { firstCol: "Data", secondCol: "Data", thirdCol: 20, fourthCol: "Data", fifthCol: 50, sixthCol: "Data", seventhCol: "Data" },
        { firstCol: "Data", secondCol: "Data", thirdCol: 20, fourthCol: "Data", fifthCol: 50, sixthCol: "Data", seventhCol: "Data" },
        { firstCol: "Data", secondCol: "Data", thirdCol: 50, fourthCol: "Data", fifthCol: 50, sixthCol: "Data", seventhCol: "Data" },
    ],
];

test("render NlpTable", () => {
    renderWithProviders(<NlpTable data={data} chatObject={ChatHistoryObject} tableError="" tableShowError={true} tableBotResponse="" />);
});

test("should update activeIndex when left arrow is clicked", () => {
    renderWithProviders(<NlpTable data={data} chatObject={ChatHistoryObject} tableBotResponse="" tableError="" tableShowError />);
    const leftArrow = screen.getByTestId("left-arrow");
    const activeIndexElement = screen.getByTestId("active-index");

    fireEvent.click(leftArrow);

    expect(activeIndexElement).toHaveTextContent("0 of 3");
});

test("should update activeIndex when right arrow is clicked", () => {
    renderWithProviders(<NlpTable data={data} chatObject={ChatHistoryObject} tableBotResponse="" tableError="" tableShowError />);
    const rightArrow = screen.getByTestId("right-arrow");
    const activeIndexElement = screen.getByTestId("active-index");

    fireEvent.click(rightArrow);

    expect(activeIndexElement).toHaveTextContent("2 of 3");
});

test("should disable left arrow when activeIndex is 0", () => {
    renderWithProviders(<NlpTable data={data} chatObject={ChatHistoryObject} tableBotResponse="" tableError="" tableShowError />);
    const leftArrow = screen.getByTestId("left-arrow");

    expect(leftArrow).toHaveClass("pointer-events-none");
});

test("should disable right arrow when activeIndex is at the last index", () => {
    renderWithProviders(<NlpTable data={data} chatObject={ChatHistoryObject} tableBotResponse="" tableError="" tableShowError />);
    const rightArrow = screen.getByTestId("right-arrow");

    fireEvent.click(rightArrow);
    fireEvent.click(rightArrow);
    fireEvent.click(rightArrow);
    fireEvent.click(rightArrow);

    expect(rightArrow).toHaveClass("pointer-events-none");
});
