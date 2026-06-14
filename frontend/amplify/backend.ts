import { defineBackend } from '@aws-amplify/backend'
import { auth } from './auth/resource'

const backend = defineBackend({ auth })

// Private app: sign in with a USERNAME (e.g. "Hose"), not an email alias.
// Users are admin-created (no public sign-up) and no email/phone verification
// is required. This is a CDK escape hatch on the generated Cognito user pool.
const { cfnUserPool } = backend.auth.resources.cfnResources
cfnUserPool.usernameAttributes = []
cfnUserPool.aliasAttributes = []
cfnUserPool.adminCreateUserConfig = {
  allowAdminCreateUserOnly: true,
}
