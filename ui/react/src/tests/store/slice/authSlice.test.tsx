import authSlice, { loginUser, logoutUser } from '../../../store/slice/authSlice';

describe('authSlice reducer', () => {
    const initialUserInfo = {
        id: 0,
        name: '',
        email: '',
        picture: '',
        last_login: '',
        date_joined: '',
    };

    it('should handle loginUser', () => {
        const initialState = {
            isAuthenticated: false,
            userInfo: initialUserInfo,
        };

        const user = {
            user_info: {
                id: 1,
                name: 'John Doe',
                email: 'john@example.com',
                picture: 'profile.jpg',
                last_login: '2022-01-01',
                date_joined: '2021-01-01',
            },
            token: 'abc123',
        };

        const action = {
            type: loginUser.type,
            payload: user,
        };

        const nextState = authSlice(initialState, action);

        expect(nextState.isAuthenticated).toBe(true);
        expect(nextState.userInfo).toEqual(user.user_info);
    });

    it('should handle logoutUser', () => {
        const initialState = {
            isAuthenticated: true,
            userInfo: {
                id: 1,
                name: 'John Doe',
                email: 'john@example.com',
                picture: 'profile.jpg',
                last_login: '2022-01-01',
                date_joined: '2021-01-01',
            },
        };

        const action = {
            type: logoutUser.type,
        };

        const nextState = authSlice(initialState, action);

        expect(nextState.isAuthenticated).toBe(false);
        expect(nextState.userInfo).toEqual(initialUserInfo);
    });
});
