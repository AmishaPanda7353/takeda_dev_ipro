import { cleanup } from "@testing-library/react"
import LoginLayout from "layouts/LoginLayout"
import renderWithProviders from "tests/utils"

afterEach(cleanup)

test("render LoginLayout", () => {
    renderWithProviders(<LoginLayout />)
})