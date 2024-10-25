import { cleanup, screen } from "@testing-library/react"
import Loadable from "components/Loadable"
import Loading from "components/Loading"
import { mockComponent } from "react-dom/test-utils"
import renderWithProviders from "tests/utils"

afterEach(cleanup)

// Mock the component to be loaded 

const MockComponent = () => <div>Sample Component </div>;

describe("Lodable", () => {
    test("render the loadable", () => {
        const LoadableComponent = Loadable(MockComponent);
        renderWithProviders(<LoadableComponent />);
        // expect(screen.getByTestId("loading-component")).toBeInTheDocument();
    })
})

