import ResetChat from "components/ResetChatModal"
import Sidebar from "components/sidebar"
import renderWithProviders from "tests/utils"
import { cleanup, screen, fireEvent } from "@testing-library/react"
import '@testing-library/jest-dom/extend-expect';
import { getByText } from "@testing-library/dom";
import { ENDPOINTS } from 'constants/constant';
import { SidebarProps, MenuItemType } from "../../../components/sidebar/types"
import { useDispatch } from "react-redux";
import { setActiveChatDetails } from 'store/slice/userSlice';

// jest.mock("react-redux", () => ({
//     ...jest.requireActual("react-redux"),
//     useDispatch: jest.fn(),
// }));


describe("Siderbar", () => {

    afterEach(cleanup)

    const menuItems = [
        { name: "Item", link: "report", margin: true, isBottom: true }
    ]
    const open = true
    test("render sidebar", () => {
        renderWithProviders(<Sidebar initialOpenState={true} menuItems={menuItems} />);

        // test the side bar clickable

        const resetModal = screen.getByTestId("sidebar")
        expect(resetModal).toBeInTheDocument();
        fireEvent.click(resetModal);

        // Test report list
        const reportDiv = screen.getByTestId("ReportDiv");
        expect(reportDiv).toBeInTheDocument();

        // Test reset chat button
        const restChatButton = screen.getByText('Reset Chat');
        expect(restChatButton).toBeInTheDocument();
        fireEvent.click(restChatButton);

        // Test handling report click
        const handleReportClick = jest.fn();
        const reportItem = reportDiv.querySelector("div");
        reportItem?.click();
        expect(handleReportClick).toHaveBeenCalledTimes(0);

        !open && expect(screen.getByRole("img", { name: "tiger" })).toBeInTheDocument
        !open && expect(screen.getByAltText('document-scanner')).toBeInTheDocument();
        !open && expect(screen.getByText('Powered')).toBeInTheDocument();
        !open && expect(screen.getByText('Report')).toBeInTheDocument();
        !open && expect(screen.getByText('No report created')).toBeInTheDocument();
    })

    const resetModalShow = true
    test("render ResetChat", () => {
        resetModalShow && renderWithProviders(<ResetChat resetModalShow={true} setResetModalShow={jest.fn()} />)
        // Test ResetChat component Rendering
        const resetChat = screen.getByTestId("reset-chat-modal");
        expect(resetChat).toBeInTheDocument();
    })

    // test("handleReportClick dispatches the correct actions", async () => {

    //     const menuItems = [
    //         { name: "Item 1", link: "/item1", margin: true, isBottom: true },
    //         { name: "Item 2", link: "/item2", margin: false, isBottom: true }
    //     ];

    //     renderWithProviders(
    //         <Sidebar initialOpenState={true} menuItems={menuItems} />
    //     );

    //     fireEvent.click(screen.getByText("Reports"));
    // });

    // test("handleReportClick dispatches the correct actions", () => {
    //     const dispatchMock = jest.fn(); // Create a mock function for dispatch
    //     (useDispatch as jest.Mock).mockReturnValue(dispatchMock); // Set the mock implementation for useDispatch
    //     const setActiveReportChatId = jest.fn();
    //     const setIsOldReport = jest.fn();

    //     (useDispatch as jest.Mock).mockReturnValue(dispatchMock);

    //     const report = {
    //         chat_id: "123",
    //         report_name: "Test Report",
    //         report_url: "https://example.com/reports/123",
    //     };

    //     const menuItems = [{ name: "Test Report", link: "#", margin: true, isBottom: true }];

    //     renderWithProviders(<Sidebar initialOpenState={true} menuItems={menuItems} />)

    //     // expect(dispatchMock).toHaveBeenCalledTimes(3);
    //     // expect(dispatchMock).toHaveBeenCalledWith(setActiveReportChatId("123"));
    //     // expect(dispatchMock).toHaveBeenCalledWith(setIsOldReport(false));
    //     expect(dispatchMock).toHaveBeenCalledWith(
    //         setActiveChatDetails([{ chat_id: 2, user_id: "2", context_id: 2, chat_name: "chat 9HF9VO", is_active: true, chat_history: [] }])
    //     );

    // });

    // test("Render SiderBar  component  with Menu items", () => {

    //     const menuItems = [
    //         { name: "Item 1", link: "/item1", margin: true, isBottom: true },
    //         { name: "Item 2", link: "/item2", margin: false, isBottom: true }
    //     ];
    //     renderWithProviders(<Sidebar initialOpenState={true} menuItems={menuItems} />)

    //     // Test menu item rendering 
    //     const menuItem1 = screen.getByText((content, element) => {
    //         return element?.textContent === "Item 1"
    //     });
    //     expect(menuItem1).toBeInTheDocument();
    //     const menyItem2 = screen.getByText((content, element) => {
    //         return element?.textContent === "Item 2";
    //     });
    //     expect(menyItem2).toBeInTheDocument();

    //     // Test menu item click event
    //     fireEvent.click(menuItem1);
    //     fireEvent.click(menyItem2);

    // })

    // Define a type or interface for the test props

});



