import { createSlice, PayloadAction } from '@reduxjs/toolkit'

export type UserObject = {
  id?: number
  name?: string
  email?: string
  picture?: string
  last_login?: string
  date_joined?: string
}

export type GroupObject = {
  id: string;
  displayName: string;
}

export type LoginAPIResponse = {
  user_info: UserObject
  token: string
  groups: Array<GroupObject>
}

export type CommonState = {
  isAuthenticated: boolean
  userInfo: UserObject
}

const initialUserInfo: UserObject = {
  id: 0,
  name: '',
  email: '',
  picture: '',
  last_login: '',
  date_joined: '',
}

const storedAuthData = JSON.parse(localStorage.getItem('auth_state') as string)

const initialState: CommonState = {
  isAuthenticated: storedAuthData ? storedAuthData.isAuthenticated : false,
  userInfo: storedAuthData ? storedAuthData.userInfo : initialUserInfo,
}
export const authSlice = createSlice({
  name: 'auth',
  initialState,
  reducers: {
    loginUser(state: CommonState, action: PayloadAction<LoginAPIResponse>) {
      state.isAuthenticated = true
      state.userInfo = action.payload.user_info
      localStorage.setItem('auth_state', JSON.stringify({isAuthenticated: state.isAuthenticated, userInfo: state.userInfo}))
    },
    logoutUser(state: CommonState) {
      // sessionStorage.removeItem('access_token')
      // sessionStorage.removeItem('user_info')
      // sessionStorage.removeItem("user_groups")
      localStorage.clear()
      sessionStorage.clear()
      sessionStorage.setItem('redirectPathBS', '/')
      state.isAuthenticated = false
      state.userInfo = initialUserInfo
    }
  },
})

export const { loginUser, logoutUser } = authSlice.actions

export default authSlice.reducer
