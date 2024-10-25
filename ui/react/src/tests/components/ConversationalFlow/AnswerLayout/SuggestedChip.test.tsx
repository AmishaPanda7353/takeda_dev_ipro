import SuggestedChip from "components/ConversationalFlow/AnswerLayout/SuggestedChip"
import renderWithProviders from "tests/utils"

test("render suggestedChip", () => {
    renderWithProviders(<SuggestedChip text="Hello" />)
})