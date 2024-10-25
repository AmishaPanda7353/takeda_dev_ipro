import { cleanup, screen } from "@testing-library/react"
import Loading from "components/Loading"
import renderWithProviders from "tests/utils"

afterEach(cleanup)

test("renders the loding ", () => {
    renderWithProviders(<Loading />)
    // expect(screen.getByTestId('loader')).toBeInTheDocument();
})