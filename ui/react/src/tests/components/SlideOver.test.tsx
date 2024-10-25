import { cleanup ,fireEvent,screen} from "@testing-library/react"
import SlideOver from "components/SlideOver"
import renderWithProviders from "tests/utils"

afterEach(cleanup)

const closeClick = jest.fn();
const closeOnClick = jest.fn();

test("render sliderOver component", () => {
    renderWithProviders(<SlideOver children={<div>HEllo</div>} closeOnClick={closeClick} />)

    const CloseBtn = screen.getAllByTestId("closeBtn")
    expect(CloseBtn).toBeInTheDocument
   renderWithProviders(<div onClick={() => closeOnClick()}></div>)
})