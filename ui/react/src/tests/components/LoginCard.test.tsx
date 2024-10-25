import { cleanup, fireEvent, screen } from "@testing-library/react"
import LoginCard from "components/LoginCard"
import renderWithProviders from "tests/utils"

afterEach(cleanup)

const mackFun = jest.fn();
const navigateMock = jest.fn();

test("render the loginCard", async () => {
    renderWithProviders(<LoginCard />)

    // const closeBtn = screen.getByRole("button", { name: 'Login' });
    // fireEvent.click(closeBtn);
    // expect(mackFun).toBeCalled();

})

// test('navigate to /insights on function call', () => {
//     // Mock the navigate function

//     jest.mock('your-navigation-library', () => ({ navigate: navigateMock }));

//     // Render the component
//     renderWithProviders(<LoginCard />);

//     // Call the routechange function
//     const routechange = screen.getByTestId('routechange-button'); // Assuming you have a button with a test ID
//     fireEvent.click(routechange);

//     // Assert that the navigate function has been called with the expected path
//     expect(navigateMock).toHaveBeenCalledWith('/insights');
// });